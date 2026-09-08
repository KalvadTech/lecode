# Changelog

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
