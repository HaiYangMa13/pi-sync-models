# pi-sync-models

A Pi package that discovers model IDs from OpenAI-compatible providers configured in `<agent-dir>/models.json`, applies intelligent context window & capability inference, upgrades existing metadata, and prunes deprecated models safely.

## Install

```sh
pi install npm:pi-sync-models
```

**If you already have `~/.pi/agent/extensions/sync-models.ts`, remove/disable that standalone copy before installing**, otherwise two copies of `/sync-models` will collide and both may run at startup. Restart Pi after installation.

## Usage

- At startup, check providers once in the background (except when `PI_OFFLINE=1`).
- `/sync-models` checks all configured providers.
- `/sync-models provider-id` checks only that provider.
- Open `/model` after a successful sync to reload Pi's model list.

Only providers with `api: "openai-responses"` or `"openai-completions"` and `baseUrl` are queried at `<baseUrl>/models`. Keys and headers may use Pi-style literal, `$ENV`, `${ENV}`, or `!command` values. This package does not read `auth.json` or discover built-in providers. Each request times out after 12 seconds. Providers that cannot be queried are left unchanged. The file's comments and unrelated configuration are preserved; writes are atomic and skipped if the file changed during the request.

**Metadata & Capabilities:**
- When discovering or syncing models, the package matches against Pi's built-in official model catalog (and heuristic rules) to assign accurate `contextWindow`, `maxTokens`, vision (`image`) modalities, and `reasoning` flags.
- Existing models with outdated or provisional metadata are upgraded.
- Models removed from the upstream endpoint are pruned safely (guarded against empty responses).
- Custom fields like `cost` on existing models are preserved.

The Pi agent directory defaults to `~/.pi/agent` and honors `PI_CODING_AGENT_DIR`. The extension executes `!command` credentials from your own `models.json` in your local shell, as Pi does. Only install and run with trusted configuration.

## Develop

```sh
npm install
npm run verify
pi -e .
```

License: MIT.
