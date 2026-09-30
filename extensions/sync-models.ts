import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { execFileSync } from "node:child_process";
import { readFile, rename, stat, unlink, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { randomUUID } from "node:crypto";
import { applyEdits, modify, parse, printParseErrorCode, type ParseError } from "jsonc-parser";

export type ProviderModel = { id: string; [key: string]: unknown };
export type ProviderConfig = {
  baseUrl?: string;
  api?: string;
  apiKey?: string;
  headers?: Record<string, string>;
  models?: ProviderModel[];
};
export type ModelsConfig = { providers: Record<string, ProviderConfig> };
export type SyncResult = { added: number; updated: number; removed: number; report: string[]; ok: boolean };

export interface BuiltinModelMeta {
  id?: string;
  contextWindow: number;
  maxTokens: number;
  input: ("text" | "image")[];
  reasoning: boolean;
  source: string;
}

export type BuiltinCatalog = Map<string, BuiltinModelMeta>;

let cachedCatalog: BuiltinCatalog | null = null;

export async function getBuiltinCatalog(): Promise<BuiltinCatalog> {
  if (cachedCatalog) return cachedCatalog;
  const catalog: BuiltinCatalog = new Map();
  try {
    const { ModelRuntime } = await import("@earendil-works/pi-coding-agent");
    const runtime = await ModelRuntime.create({ refreshOnCreate: false, modelsPath: null });
    for (const m of runtime.getModels()) {
      const meta: BuiltinModelMeta = {
        id: m.id,
        contextWindow: m.contextWindow || 128000,
        maxTokens: m.maxTokens || 16384,
        input: m.input && Array.isArray(m.input) ? (m.input.slice() as ("text" | "image")[]) : ["text"],
        reasoning: Boolean(m.reasoning),
        source: "catalog",
      };
      const lower = m.id.toLowerCase();
      catalog.set(lower, meta);
      const tail = lower.split(/[/:]/).pop();
      if (tail && !catalog.has(tail)) catalog.set(tail, meta);
    }
  } catch {
    // If ModelRuntime cannot be initialized, fallback gracefully
  }
  cachedCatalog = catalog;
  return catalog;
}

function normalizeModelId(rawId: string): string {
  let s = rawId.toLowerCase().trim();
  s = s.replace(/^[a-z0-9_-]+\//, "");
  s = s.replace(/^[a-z0-9_-]+:/, "");
  return s;
}

function stripSuffixes(id: string): string {
  let s = id;
  s = s.replace(/-\d{4,8}$/, "");
  const suffixes = [
    /-thinking(?:-[a-z0-9_-]+)*$/,
    /-(?:high|low|medium|max|xhigh|minimal)$/,
    /-(?:fast|slow)$/,
    /-non-reasoning$/,
    /-reasoning$/,
    /-search$/,
    /-online$/,
    /-preview$/,
    /-latest$/,
  ];
  let changed = true;
  while (changed) {
    changed = false;
    for (const re of suffixes) {
      if (re.test(s)) {
        s = s.replace(re, "");
        changed = true;
      }
    }
  }
  return s;
}

export function resolveModelMeta(rawId: string, catalog?: BuiltinCatalog): BuiltinModelMeta {
  const cat = catalog ?? cachedCatalog;
  const lower = rawId.toLowerCase().trim();
  const normalized = normalizeModelId(rawId);
  const stripped = stripSuffixes(normalized);

  const candidateKeys = [
    lower,
    normalized,
    stripped,
    stripSuffixes(lower),
  ];

  let matched: BuiltinModelMeta | undefined;
  if (cat) {
    for (const k of candidateKeys) {
      matched = cat.get(k);
      if (matched) break;
    }
  }

  const hasReasoningMarker = /(?:thinking|reasoning|reasoner|thought)/i.test(rawId);
  const reasoning = hasReasoningMarker || Boolean(matched?.reasoning);

  if (matched) {
    return {
      id: matched.id,
      contextWindow: matched.contextWindow,
      maxTokens: matched.maxTokens,
      input: matched.input.slice(),
      reasoning,
      source: `catalog:${matched.id ?? ""}`,
    };
  }

  let contextWindow = 128000;
  let maxTokens = 16384;
  let source = "fallback";
  const input: ("text" | "image")[] = ["text"];

  if (/(?:image|vision|claude|gemini|gpt-4|gpt-5|gpt-6|omni|qwen-vl)/i.test(rawId)) {
    input.push("image");
    source = "rule";
  }

  if (/gemini/i.test(rawId)) {
    contextWindow = 1000000;
    maxTokens = 65536;
    source = "rule";
  } else if (/claude/i.test(rawId)) {
    if (/(?:opus-5|sonnet-5|opus-4|sonnet-4)/i.test(rawId)) {
      contextWindow = 1000000;
    } else {
      contextWindow = 200000;
    }
    maxTokens = 64000;
    source = "rule";
  } else if (/deepseek/i.test(rawId)) {
    if (/(?:v4|r2)/i.test(rawId)) {
      contextWindow = 1000000;
    } else {
      contextWindow = 128000;
    }
    maxTokens = 32768;
    source = "rule";
  } else if (/grok/i.test(rawId)) {
    contextWindow = 500000;
    maxTokens = 128000;
    source = "rule";
  } else if (/(?:1m|1000k)/i.test(rawId)) {
    contextWindow = 1000000;
    source = "rule";
  } else if (/2m/i.test(rawId)) {
    contextWindow = 2000000;
    source = "rule";
  } else if (/200k/i.test(rawId)) {
    contextWindow = 200000;
    source = "rule";
  } else if (/256k/i.test(rawId)) {
    contextWindow = 262144;
    source = "rule";
  }

  return {
    contextWindow,
    maxTokens,
    input,
    reasoning,
    source,
  };
}

export function newModel(id: string, catalog?: BuiltinCatalog): ProviderModel {
  const meta = resolveModelMeta(id, catalog);
  return {
    id,
    name: id,
    reasoning: meta.reasoning,
    input: meta.input,
    contextWindow: meta.contextWindow,
    maxTokens: meta.maxTokens,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
  };
}

export function patchModel(existing: ProviderModel, catalog?: BuiltinCatalog): { model: ProviderModel; changed: boolean } {
  const meta = resolveModelMeta(existing.id, catalog);
  let changed = false;

  const currentCw = existing.contextWindow as number | undefined;
  const currentMaxTokens = existing.maxTokens as number | undefined;
  const currentReasoning = existing.reasoning as boolean | undefined;
  const currentInput = Array.isArray(existing.input) ? (existing.input as ("text" | "image")[]) : [];

  let nextCw = currentCw;
  if (meta.source.startsWith("catalog") || currentCw === 128000 || currentCw === 1000000 || currentCw === undefined) {
    nextCw = meta.contextWindow;
  }

  let nextMaxTokens = currentMaxTokens;
  if (meta.source.startsWith("catalog") || currentMaxTokens === 16384 || currentMaxTokens === undefined) {
    nextMaxTokens = meta.maxTokens;
  }

  const nextReasoning = Boolean(currentReasoning || meta.reasoning);

  let nextInput = currentInput;
  if (meta.source.startsWith("catalog") || currentInput.length === 0) {
    nextInput = meta.input;
  } else {
    const inputSet = new Set(currentInput);
    for (const item of meta.input) inputSet.add(item);
    nextInput = Array.from(inputSet) as ("text" | "image")[];
  }

  const inputChanged =
    currentInput.length !== nextInput.length ||
    currentInput.some((v, i) => v !== nextInput[i]);

  if (
    currentCw !== nextCw ||
    currentMaxTokens !== nextMaxTokens ||
    currentReasoning !== nextReasoning ||
    inputChanged
  ) {
    changed = true;
  }

  return {
    changed,
    model: {
      ...existing,
      contextWindow: nextCw,
      maxTokens: nextMaxTokens,
      reasoning: nextReasoning,
      input: nextInput,
    },
  };
}

export function isChatCandidate(id: string): boolean {
  return !/(?:image|imagine|video|embedding|embed|whisper|transcri|speech|tts|audio|realtime|moderation)/i.test(id);
}

function parseConfig(text: string): ModelsConfig {
  const errors: ParseError[] = [];
  const config: unknown = parse(text, errors, { allowTrailingComma: true });
  if (errors.length) throw new Error(`Invalid models.json: ${printParseErrorCode(errors[0].error)}`);
  if (!config || typeof config !== "object" || Array.isArray(config) ||
      !("providers" in config) || !config.providers || typeof config.providers !== "object" || Array.isArray(config.providers)) {
    throw new Error("Invalid models.json: expected providers object");
  }
  return config as ModelsConfig;
}

async function fetchIds(provider: ProviderConfig): Promise<string[]> {
  if (!provider.baseUrl || !provider.api?.startsWith("openai-")) {
    throw new Error("not an OpenAI-compatible provider with a baseUrl");
  }
  const base = new URL(provider.baseUrl);
  if (!["http:", "https:"].includes(base.protocol) || base.username || base.password) throw new Error("Invalid provider baseUrl");
  const url = new URL(base.toString());
  url.pathname = `${url.pathname.replace(/\/+$/, "").replace(/\/models$/, "")}/models`;
  const headers = new Headers({ Accept: "application/json" });
  try {
    for (const [name, value] of Object.entries(provider.headers ?? {})) headers.set(name, resolveSecret(value));
    if (!headers.has("Authorization") && provider.apiKey) headers.set("Authorization", `Bearer ${resolveSecret(provider.apiKey)}`);
  } catch {
    throw new Error("Could not resolve provider credentials");
  }
  if (!headers.has("Authorization") && !provider.apiKey) throw new Error("No apiKey or Authorization header in models.json");
  // Only model IDs are used. No provider response or credential is logged or persisted.
  const response = await fetch(url, { headers, signal: AbortSignal.timeout(12000) });
  if (!response.ok) throw new Error(`Model endpoint returned HTTP ${response.status}`);
  const payload: unknown = await response.json();
  if (!payload || typeof payload !== "object" || !("data" in payload) || !Array.isArray(payload.data)) {
    throw new Error("Model endpoint did not return a data array");
  }
  const ids = payload.data.map((entry: unknown) =>
    entry && typeof entry === "object" && "id" in entry ? entry.id : undefined);
  if (ids.some((id: unknown) => typeof id !== "string" || !id.trim())) throw new Error("Model endpoint returned an invalid ID");
  return [...new Set(ids as string[])].filter(isChatCandidate);
}

export function agentModelsPath(): string {
  return join(process.env.PI_CODING_AGENT_DIR || join(homedir(), ".pi", "agent"), "models.json");
}

// Pi supports literal, environment and command credentials. Never include the resolved value in errors.
export function resolveSecret(value: string): string {
  if (value.startsWith("!")) {
    return execFileSync("/bin/sh", ["-c", value.slice(1)], {
      encoding: "utf8", timeout: 5000, maxBuffer: 65536, stdio: ["ignore", "pipe", "ignore"],
    }).trim();
  }
  return value.replace(/\$\$|\$!|\$\{([A-Za-z_][A-Za-z_0-9]*)\}|\$([A-Za-z_][A-Za-z_0-9]*)/g,
    (matched, braced: string | undefined, plain: string | undefined) => {
      if (matched === "$$") return "$";
      if (matched === "$!") return "!";
      const resolved = process.env[(braced ?? plain)!];
      if (!resolved) throw new Error("Missing environment credential");
      return resolved;
    });
}

/** Syncs newly discovered IDs, updates existing metadata, and prunes deprecated models. */
export async function syncModels(path = agentModelsPath(), target?: string): Promise<SyncResult> {
  const original = await readFile(path, "utf8");
  const config = parseConfig(original);
  const providerIds = target ? [target] : Object.keys(config.providers);
  if (target && !Object.hasOwn(config.providers, target)) return { added: 0, updated: 0, removed: 0, ok: false, report: [`Unknown provider: ${target}`] };
  let text = original;
  let added = 0;
  let updated = 0;
  let removed = 0;
  let ok = true;
  const report: string[] = [];
  const catalog = await getBuiltinCatalog();
  for (const id of providerIds) {
    const provider = config.providers[id];
    try {
      const remote = await fetchIds(provider);
      const existing = provider.models ?? [];
      if (!Array.isArray(existing) || existing.some((model) => !model || typeof model.id !== "string")) {
        throw new Error("Invalid local models array");
      }
      if (existing.length > 0 && remote.length === 0) {
        throw new Error("Model endpoint returned an empty model list; skipping to prevent accidental deletion");
      }

      const remoteSet = new Set(remote);
      const seen = new Set<string>();
      let providerUpdated = 0;
      let providerRemoved = 0;
      const merged: ProviderModel[] = [];

      for (const model of existing) {
        if (!remoteSet.has(model.id)) {
          providerRemoved++;
          continue;
        }
        seen.add(model.id);
        const { model: patched, changed } = patchModel(model, catalog);
        if (changed) providerUpdated++;
        merged.push(patched);
      }

      const fresh = remote.filter((modelId) => !seen.has(modelId));
      for (const modelId of fresh) {
        merged.push(newModel(modelId, catalog));
      }

      if (fresh.length > 0 || providerUpdated > 0 || providerRemoved > 0) {
        text = applyEdits(text, modify(text, ["providers", id, "models"], merged, {
          formattingOptions: { insertSpaces: true, tabSize: 2, eol: "\n" },
        }));
        added += fresh.length;
        updated += providerUpdated;
        removed += providerRemoved;
      }
      report.push(`${id}: +${fresh.length} added, ${providerUpdated} updated, -${providerRemoved} removed`);
    } catch (error) {
      ok = false;
      // Only our own fixed errors; network errors may embed sensitive URLs, so omit their details.
      const safe = error instanceof Error && /^(not an OpenAI|Invalid provider|Could not resolve|No apiKey|Model endpoint|Invalid local)/.test(error.message)
        ? error.message : "request or parsing failed";
      report.push(`${id}: ${safe}`);
    }
  }
  if (added || updated || removed) {
    // Refuse to clobber an edit made during the network request; never write partial/corrupt JSON.
    parseConfig(text);
    if (await readFile(path, "utf8") !== original) throw new Error("models.json changed during sync; retry");
    const temp = join(dirname(path), `.models.json.${randomUUID()}.tmp`);
    try {
      const mode = (await stat(path)).mode & 0o777;
      await writeFile(temp, text, { encoding: "utf8", mode, flag: "wx" });
      await rename(temp, path);
    } finally {
      await unlink(temp).catch(() => {});
    }
  }
  return { added, updated, removed, report, ok };
}

export default function (pi: ExtensionAPI) {
  let queue: Promise<void> = Promise.resolve();
  const run = (target?: string) => {
    const result = queue.then(() => syncModels(agentModelsPath(), target));
    queue = result.then(() => {}, () => {});
    return result;
  };
  pi.registerCommand("sync-models", {
    description: "Discover and sync OpenAI-compatible models from models.json provider endpoints",
    handler: async (args, ctx) => {
      try {
        const result = await run(args.trim() || undefined);
        ctx.ui.notify(`${result.report.join("\n")}\nAdded ${result.added}, updated ${result.updated}, removed ${result.removed}. Open /model to reload.`, result.ok ? "info" : "warning");
      } catch {
        ctx.ui.notify("Model sync failed; models.json was not changed (check file and retry).", "error");
      }
    },
  });
  pi.on("session_start", (event, ctx) => {
    if (event.reason !== "startup" || process.env.PI_OFFLINE === "1") return;
    void run().then((result) => {
      if (result.added || result.updated || result.removed) {
        ctx.ui.notify(`Synced ${result.added} new, updated ${result.updated}, removed ${result.removed} models. Open /model to reload.`, "info");
      }
    }).catch(() => {});
  });
}
