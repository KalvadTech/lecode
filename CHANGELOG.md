# Changelog

## [0.3.0](https://github.com/KalvadTech/lecode/compare/v0.2.0...v0.3.0) (2026-10-10)


### ⚠ BREAKING CHANGES

* remove the built-in exa and context7 MCP servers; configure servers under [mcp.servers] instead ([898c835](https://github.com/KalvadTech/lecode/commit/898c835))
* remove the --version flag and the version from the welcome screen ([eebd654](https://github.com/KalvadTech/lecode/commit/eebd654))

### Features

* persistent worker subagents with supervision ([248bb2a](https://github.com/KalvadTech/lecode/commit/248bb2a))
* source-linked hybrid memory with safe forgetting ([c204c02](https://github.com/KalvadTech/lecode/commit/c204c02))
* live agent roster and /runs detail panel ([49036f0](https://github.com/KalvadTech/lecode/commit/49036f0))
* argument pickers for slash commands (/model, /resume, …) ([20cb421](https://github.com/KalvadTech/lecode/commit/20cb421))
* pick ask_user answers with the arrow-key picker ([0fb73ad](https://github.com/KalvadTech/lecode/commit/0fb73ad))
* render assistant messages as markdown at stream end ([068cc98](https://github.com/KalvadTech/lecode/commit/068cc98))
* integrate with herdr ([5c1a958](https://github.com/KalvadTech/lecode/commit/5c1a958))
* show average token speed in the statusline ([aade96e](https://github.com/KalvadTech/lecode/commit/aade96e))
* import MCP servers from opencode in the setup wizard ([d3222df](https://github.com/KalvadTech/lecode/commit/d3222df))
* gradient block welcome logo by Kalvad ([0761dc3](https://github.com/KalvadTech/lecode/commit/0761dc3))
* Shift+Enter inserts a newline via terminal key modes ([90506f0](https://github.com/KalvadTech/lecode/commit/90506f0))
* run shell commands through the detected user shell ([d2a0f2b](https://github.com/KalvadTech/lecode/commit/d2a0f2b))
* bound headless runs by cost and execution time ([76c9ac6](https://github.com/KalvadTech/lecode/commit/76c9ac6))
* JSON output for noninteractive runs ([97a6968](https://github.com/KalvadTech/lecode/commit/97a6968))
* per-run reasoning effort and HTTP headers ([1aca671](https://github.com/KalvadTech/lecode/commit/1aca671))
* distinguish provider failures in scripted runs ([ca1774a](https://github.com/KalvadTech/lecode/commit/ca1774a))
* gradient-colored model table for /models and onboarding ([67a69b5](https://github.com/KalvadTech/lecode/commit/67a69b5))
* four-stop log-scaled cost ramp in the model table ([b68769c](https://github.com/KalvadTech/lecode/commit/b68769c))
* purple context ramp in the model table ([5b79fbd](https://github.com/KalvadTech/lecode/commit/5b79fbd))
* anchor the working directory in the system prompt ([24264a9](https://github.com/KalvadTech/lecode/commit/24264a9))
* dependency pinning, writing style, and commit rules in the base prompts ([33575cf](https://github.com/KalvadTech/lecode/commit/33575cf))
* /mcp enable|disable to toggle servers in-session ([8b3fbf2](https://github.com/KalvadTech/lecode/commit/8b3fbf2))

### Bug Fixes

* name EOF/KI exits on stderr ([ea8eb73](https://github.com/KalvadTech/lecode/commit/ea8eb73))
* fix the setup wizard with custom providers ([c851e7c](https://github.com/KalvadTech/lecode/commit/c851e7c))
* make compaction omission-free and refresh the live system prompt ([5c6912e](https://github.com/KalvadTech/lecode/commit/5c6912e))
* make worker submission idempotent ([6b5ca03](https://github.com/KalvadTech/lecode/commit/6b5ca03))
* close worker review gaps ([9de3559](https://github.com/KalvadTech/lecode/commit/9de3559))
* preserve model calls across worker updates ([4739ae6](https://github.com/KalvadTech/lecode/commit/4739ae6))
* retry concurrent WAL initialization for facts ([9810693](https://github.com/KalvadTech/lecode/commit/9810693))
* classify malformed response encodings as stream failures ([5e0067d](https://github.com/KalvadTech/lecode/commit/5e0067d))
* reap the shell before killing surviving process group members ([be7c2ea](https://github.com/KalvadTech/lecode/commit/be7c2ea))
* bound capture even with a one-byte output cap ([732804f](https://github.com/KalvadTech/lecode/commit/732804f))
* xterm input: preserve shifted characters, Unicode, and shortcuts across key modes ([094ab96](https://github.com/KalvadTech/lecode/commit/094ab96))
* allow idle Ctrl+C to exit completed worker views ([7231b94](https://github.com/KalvadTech/lecode/commit/7231b94))
* capitalize Pierre in the reviewer prompt ([d03cc5f](https://github.com/KalvadTech/lecode/commit/d03cc5f))

### Performance Improvements

* reduce headless memory usage ([a3edc37](https://github.com/KalvadTech/lecode/commit/a3edc37))

### Documentation

* add AGENTS.md repository guidance ([57060c2](https://github.com/KalvadTech/lecode/commit/57060c2))
* focus memory documentation on lecode ([5a9ab8d](https://github.com/KalvadTech/lecode/commit/5a9ab8d))
* README covers the /models table and /mcp enable|disable ([77bb3bc](https://github.com/KalvadTech/lecode/commit/77bb3bc))

### Continuous Integration

* remove the release-please automation ([f9c7e42](https://github.com/KalvadTech/lecode/commit/f9c7e42))
* bump actions/checkout, setup-uv, and the pinned uv version ([3f83fb1](https://github.com/KalvadTech/lecode/commit/3f83fb1))


## [0.2.0](https://github.com/KalvadTech/lecode/compare/v0.1.0...v0.2.0) (2026-09-08)


### Features

* 15 lifecycle hook events (wire Stop/UserPromptSubmit/Session*, add 7 new) ([a30f914](https://github.com/KalvadTech/lecode/commit/a30f91489cab8a36066060319d33d265cef828c4))
* ask_user tool for structured mid-turn questions ([5f0cea3](https://github.com/KalvadTech/lecode/commit/5f0cea3b7f54290313e2c41aed3079fcd567e03d))
* automatic context compaction near the context window ([22c5a3e](https://github.com/KalvadTech/lecode/commit/22c5a3ed502769eacd5e51f5033819caae64d987))
* background tasks for bash and subagents (tasks_* tools, /tasks) ([ebc3663](https://github.com/KalvadTech/lecode/commit/ebc36633cd592a8f9648254a3b62d10a968c494e))
* desktop notifications (osascript/notify-send) alongside audio ([fe5715c](https://github.com/KalvadTech/lecode/commit/fe5715c9900efdacf9d6fbb417378a4c4542a739))
* list only models from the last 3 months in the setup wizard ([a272fc6](https://github.com/KalvadTech/lecode/commit/a272fc64bf4e3c0cfef4bed4a19f84218fe88869))
* MCP SSE transport and OAuth with /mcp login|logout ([4d4be2c](https://github.com/KalvadTech/lecode/commit/4d4be2c097d6d803ba185dc3de7d08e7f7cc1d04))
* OAuth 2.1 for streamable-HTTP MCP servers (/mcp auth) ([3c901b5](https://github.com/KalvadTech/lecode/commit/3c901b50d95d0d17764def54a16353cf1f475413))
* OAuth 2.1 for streamable-HTTP MCP servers (/mcp auth) ([ceaff99](https://github.com/KalvadTech/lecode/commit/ceaff9946f16f1be610630982b451ceca26d20e2))
* refuse /pierre on without a dedicated reviewer model ([4712f53](https://github.com/KalvadTech/lecode/commit/4712f53b5b1afcaaadf94e2fe4d85a79dd9a2d2e))
* show reasoning-level override in the statusline ([c84c3f6](https://github.com/KalvadTech/lecode/commit/c84c3f62bac041b314289faf3764c9864ce29890))
* show reasoning-level override in the statusline ([eef6f33](https://github.com/KalvadTech/lecode/commit/eef6f338041ac40233773f1c5094362b9f938c5a))
* slash-command dropdown with a themed panel ([cffa284](https://github.com/KalvadTech/lecode/commit/cffa284b4ed5f675332af488cd3118fba89e9cb5))
* slash-command dropdown with a themed panel ([38c3e02](https://github.com/KalvadTech/lecode/commit/38c3e020ef6f979a4f570de220bfe4e32bd01128))
* themed picker panel for @ and . menus too ([0644000](https://github.com/KalvadTech/lecode/commit/06440002c9e5eb0bed0728571eb6779a8e6bf73c))


### Bug Fixes

* . personas picker owns input start; cap path completions ([84228da](https://github.com/KalvadTech/lecode/commit/84228da095a16f20e834391b6e6dfdb4190041ea))
* export sessions to &lt;config_dir&gt;/exports by default ([b5fa213](https://github.com/KalvadTech/lecode/commit/b5fa213152b3c13a4ee4f4e68de6a55351e108ac))
* print the tool result marker on its own line after the output ([e23245f](https://github.com/KalvadTech/lecode/commit/e23245fef63fa0197d43c98b4c2f33bb2a237b37))
* show the OAuth authorization URL and quiet the startup flow traceback ([8bd5059](https://github.com/KalvadTech/lecode/commit/8bd505937cb67e9177a453b11f02d8d83256b0e9))


### Documentation

* install from the git repo in the README ([ffb5e77](https://github.com/KalvadTech/lecode/commit/ffb5e7783737910546ba34427926f68626e8aa66))
* README covers all features (background tasks, ask_user, hooks, pickers, personas, doctor) ([a8bc60f](https://github.com/KalvadTech/lecode/commit/a8bc60f49e79def67c9808210d29bd6852d3cfbb))
