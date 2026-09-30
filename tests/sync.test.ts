import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, before, test } from "node:test";
import { parse } from "jsonc-parser";
import { isChatCandidate, newModel, patchModel, resolveModelMeta, resolveSecret, syncModels } from "../extensions/sync-models.js";

let dir: string;
let file: string;
const originalFetch = globalThis.fetch;
before(async () => {
  dir = await mkdtemp(join(tmpdir(), "pi-sync-models-"));
  file = join(dir, "models.json");
});
after(async () => {
  globalThis.fetch = originalFetch;
  await rm(dir, { recursive: true, force: true });
});

const fixture = `{
  // existing secret and metadata must be preserved
  "providers": {
    "proxy": {
      "baseUrl": "https://example.test/v1",
      "api": "openai-responses",
      "apiKey": "$SYNC_TEST_KEY",
      "models": [{ "id": "old", "reasoning": true, "contextWindow": 42000 }]
    },
    "other": { "baseUrl": "https://example.test/v1", "api": "anthropic-messages" }
  }
}\n`;

test("syncs new IDs once, skips unsupported providers, keeps comments/metadata and does not persist resolved secrets", async () => {
  process.env.SYNC_TEST_KEY = "sensitive-secret";
  await writeFile(file, fixture);
  const calls: string[] = [];
  globalThis.fetch = (async (input, init) => {
    calls.push(String(input));
    assert.equal(new Headers(init?.headers).get("Authorization"), "Bearer sensitive-secret");
    return new Response(JSON.stringify({ data: [{ id: "old" }, { id: "new" }, { id: "new" }, { id: "gpt-image-2" }] }), { status: 200 });
  }) as typeof fetch;
  const first = await syncModels(file);
  assert.equal(first.added, 1);
  assert.equal(first.ok, false); // other is not OpenAI-compatible
  assert.deepEqual(calls, ["https://example.test/v1/models"]);
  const content = await readFile(file, "utf8");
  assert.match(content, /\/\/ existing secret/);
  assert.doesNotMatch(content, /sensitive-secret/);
  const config = parse(content);
  assert.equal(config.providers.proxy.models[0].contextWindow, 42000);
  assert.deepEqual(config.providers.proxy.models[1].input, ["text"]);
  assert.equal(config.providers.proxy.models[1].reasoning, false);
  const second = await syncModels(file, "proxy");
  assert.equal(second.added, 0);
  assert.equal(await readFile(file, "utf8"), content);
});

test("does not write on malformed or failed response, or unknown target", async () => {
  await writeFile(file, fixture);
  globalThis.fetch = (async () => new Response(JSON.stringify({ data: [{ foo: "bar" }] }), { status: 200 })) as typeof fetch;
  assert.deepEqual(await syncModels(file, "proxy"), { added: 0, updated: 0, removed: 0, ok: false, report: ["proxy: Model endpoint returned an invalid ID"] });
  assert.equal(await readFile(file, "utf8"), fixture);
  assert.equal((await syncModels(file, "missing")).ok, false);
  await writeFile(file, '{"providers":');
  await assert.rejects(syncModels(file), /Invalid models.json/);
});

test("credential interpolation, model filters", () => {
  process.env.SYNC_TEST_KEY = "xyz";
  assert.equal(resolveSecret("Bearer ${SYNC_TEST_KEY}"), "Bearer xyz");
  assert.equal(resolveSecret("$$SYNC_TEST_KEY"), "$SYNC_TEST_KEY");
  assert.throws(() => resolveSecret("$SYNC_TEST_KEY_MISSING"), /Missing environment credential/);
  assert.equal(isChatCandidate("text-embedding-3"), false);
  assert.equal(isChatCandidate("gpt-5.6"), true);
});

test("infers contextWindow and vision capabilities from Pi model catalog and heuristic rules", async () => {
  // 1. Built-in catalog match
  const claudeSonnet = newModel("claude-sonnet-4-5");
  assert.equal(claudeSonnet.contextWindow, 1000000);
  assert.deepEqual(claudeSonnet.input, ["text", "image"]);
  assert.equal(claudeSonnet.reasoning, true);

  // 2. Stripping proxy prefix (cursor/) and variant suffixes (-thinking, -xhigh, -fast)
  const cursorClaude = newModel("cursor/claude-opus-5-thinking-xhigh-fast");
  assert.equal(cursorClaude.contextWindow, 1000000);
  assert.deepEqual(cursorClaude.input, ["text", "image"]);
  assert.equal(cursorClaude.reasoning, true);

  // 3. Fallback rule for unknown models
  const unknown1M = newModel("custom-model-1m");
  assert.equal(unknown1M.contextWindow, 1000000);
  assert.deepEqual(unknown1M.input, ["text"]);

  const unknownGemini = newModel("gemini-unreleased-custom");
  assert.equal(unknownGemini.contextWindow, 1000000);
  assert.deepEqual(unknownGemini.input, ["text", "image"]);

  const fallbackModel = newModel("totally-unknown-model");
  assert.equal(fallbackModel.contextWindow, 128000);
  assert.deepEqual(fallbackModel.input, ["text"]);
  assert.equal(fallbackModel.reasoning, false);
});

test("patches and overwrites existing models with upgraded contextWindow and capabilities", async () => {
  // Existing model with outdated 128k contextWindow and text-only input
  const outdatedModel = {
    id: "claude-sonnet-4-5",
    name: "claude-sonnet-4-5",
    contextWindow: 128000,
    maxTokens: 16384,
    reasoning: false,
    input: ["text"],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
  };

  const { model: patched, changed } = patchModel(outdatedModel);
  assert.equal(changed, true);
  assert.equal(patched.contextWindow, 1000000);
  assert.deepEqual(patched.input, ["text", "image"]);
  assert.equal(patched.reasoning, true);
  // Preserves existing custom fields like cost
  assert.deepEqual(patched.cost, outdatedModel.cost);

  // Syncing a file with existing outdated models will upgrade them
  const updateFixture = `{
  "providers": {
    "proxy": {
      "baseUrl": "https://example.test/v1",
      "api": "openai-responses",
      "apiKey": "test-key",
      "models": [
        { "id": "claude-sonnet-4-5", "contextWindow": 128000, "input": ["text"] }
      ]
    }
  }
}\n`;
  await writeFile(file, updateFixture);
  globalThis.fetch = (async () => {
    return new Response(JSON.stringify({ data: [{ id: "claude-sonnet-4-5" }] }), { status: 200 });
  }) as typeof fetch;

  const res = await syncModels(file, "proxy");
  assert.equal(res.ok, true);
  assert.equal(res.added, 0);
  assert.equal(res.updated, 1);
  assert.equal(res.removed, 0);

  const updatedConfig = parse(await readFile(file, "utf8"));
  assert.equal(updatedConfig.providers.proxy.models[0].contextWindow, 1000000);
  assert.deepEqual(updatedConfig.providers.proxy.models[0].input, ["text", "image"]);
});

test("prunes deprecated models not present in upstream endpoint and guards against empty arrays", async () => {
  const pruneFixture = `{
  "providers": {
    "proxy": {
      "baseUrl": "https://example.test/v1",
      "api": "openai-responses",
      "apiKey": "test-key",
      "models": [
        { "id": "model-stay", "contextWindow": 128000 },
        { "id": "model-deprecated", "contextWindow": 128000 }
      ]
    }
  }
}\n`;
  await writeFile(file, pruneFixture);

  // 1. Guard against empty remote model array
  globalThis.fetch = (async () => {
    return new Response(JSON.stringify({ data: [] }), { status: 200 });
  }) as typeof fetch;

  const emptyRes = await syncModels(file, "proxy");
  assert.equal(emptyRes.ok, false);
  assert.match(emptyRes.report[0], /skipping to prevent accidental deletion/);
  // Ensure nothing was deleted when remote returned empty
  const unchanged = parse(await readFile(file, "utf8"));
  assert.equal(unchanged.providers.proxy.models.length, 2);

  // 2. Upstream returns model-stay and model-new, dropping model-deprecated
  globalThis.fetch = (async () => {
    return new Response(JSON.stringify({
      data: [{ id: "model-stay" }, { id: "model-new" }]
    }), { status: 200 });
  }) as typeof fetch;

  const syncRes = await syncModels(file, "proxy");
  assert.equal(syncRes.ok, true);
  assert.equal(syncRes.added, 1);
  assert.equal(syncRes.removed, 1);

  const finalConfig = parse(await readFile(file, "utf8"));
  const remainingIds = finalConfig.providers.proxy.models.map((m: { id: string }) => m.id);
  assert.deepEqual(remainingIds, ["model-stay", "model-new"]);
});


