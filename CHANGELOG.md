# Changelog

## [0.3.0](https://github.com/KalvadTech/lecode/compare/v0.2.0...v0.3.0) (2026-09-25)


### Features

* add durable memory and natural preference learning ([320e0ff](https://github.com/KalvadTech/lecode/commit/320e0fff7b47195fd851ea3acbd30b3aa01092c7))
* add persistent worker subagents ([248bb2a](https://github.com/KalvadTech/lecode/commit/248bb2a8347bdd96d9756bf7a20f1f16ac232992))
* add persistent worker supervision ([137f81d](https://github.com/KalvadTech/lecode/commit/137f81d6a9054feae7ac94e7399459ede5b881bd))
* add source-linked hybrid memory and safe forgetting ([c204c02](https://github.com/KalvadTech/lecode/commit/c204c02e8d5fe9f3e4f90eb8cd8dee53ff4babe9))
* argument pickers for slash commands (/model, /resume, …) ([2419ec5](https://github.com/KalvadTech/lecode/commit/2419ec5e18c08f0e15c36c1fb17fcaec098de698))
* complete persistent worker supervision ([d48eacc](https://github.com/KalvadTech/lecode/commit/d48eacc64879fcc75487102d218c79fcac526500))
* import MCP servers from opencode in the setup wizard ([a8687c9](https://github.com/KalvadTech/lecode/commit/a8687c9453121a77b6d1bf8c0b5a6f88cedf4aaa))
* import MCP servers from opencode in the setup wizard ([d3222df](https://github.com/KalvadTech/lecode/commit/d3222dfc1bc7ae34abe2bfb088b317928c727c1b))
* integrate with herdr ([5daa144](https://github.com/KalvadTech/lecode/commit/5daa1441f542c375c19672505805d78e0c049c0c))
* integrate with herdr ([5c1a958](https://github.com/KalvadTech/lecode/commit/5c1a95833e705ecd5b102ad13562129f7eac2559))
* live agent roster, /runs detail panel, concise tool lines ([49036f0](https://github.com/KalvadTech/lecode/commit/49036f05bc081e818b0190141b462850363acfc5))
* pick ask_user answers with the arrow-key picker ([f111459](https://github.com/KalvadTech/lecode/commit/f111459e73e9979eb8b3d9941520f181f9b1dc83))
* pick ask_user answers with the arrow-key picker ([0fb73ad](https://github.com/KalvadTech/lecode/commit/0fb73ad25d8af0eddc55547ba2cff7730706f158))
* render markdown after stream end ([55ecdfd](https://github.com/KalvadTech/lecode/commit/55ecdfd88ca4840ce2511fa115b3f536e0262e61))
* render markdown after stream end ([068cc98](https://github.com/KalvadTech/lecode/commit/068cc98c9e5e2c98248e7041474f293c0aaf320f))
* show average token speed in statusline ([f9b40a6](https://github.com/KalvadTech/lecode/commit/f9b40a634721dbafbf9e8f978c474089cbdb404e))
* show average token speed in statusline ([aade96e](https://github.com/KalvadTech/lecode/commit/aade96ee9bbf32d74f4c4301e6cab7e6575c7fee))
* tag subagent runs with identity and persist bounded activity ([e36a36f](https://github.com/KalvadTech/lecode/commit/e36a36f4846b9820b0cafa1ee0bdbb50c5f31ab8))


### Bug Fixes

* close worker review gaps ([9de3559](https://github.com/KalvadTech/lecode/commit/9de35591a8735a7801557651ddc84c39b71b2a1e))
* learn natural preferences and expose extraction diagnostics ([29a9d12](https://github.com/KalvadTech/lecode/commit/29a9d12335375850b1d258555825f578089c155a))
* make compaction omission-free and refresh the live system prompt ([5c6912e](https://github.com/KalvadTech/lecode/commit/5c6912e48042c396d8f53377b88cd3389acdfdff))
* make worker submission idempotent ([6b5ca03](https://github.com/KalvadTech/lecode/commit/6b5ca03b5cec90ef2d089bd274f2785794a06d95))
* preserve model calls across worker updates ([4739ae6](https://github.com/KalvadTech/lecode/commit/4739ae6ca940ce0c0e69bf7601c1d4e759aab287))
* render the roster window as ANSI, not Rich Text ([aba7128](https://github.com/KalvadTech/lecode/commit/aba712817088da7df427bfe0c175fee29c746e5e))
* retry concurrent WAL initialization for facts ([9810693](https://github.com/KalvadTech/lecode/commit/9810693468696107a59cbb3288ce52728706f87f))
* setup not working with custom provider ([f328f4a](https://github.com/KalvadTech/lecode/commit/f328f4ab85acc63175c4e0a62979f09b9ba34634))
* setup not working with custom provider ([c851e7c](https://github.com/KalvadTech/lecode/commit/c851e7c59b8ae8ba646714f67b9bc31abf294721))


### Documentation

* add AGENTS.md repository guidance ([5749e96](https://github.com/KalvadTech/lecode/commit/5749e96e5e82383af8a8b8963ce6ce2121dc802f))
* add AGENTS.md repository guidance ([57060c2](https://github.com/KalvadTech/lecode/commit/57060c21f09595a547ed4f3c60c528d2f37c99df))
* add memory comparison and upgrade plan ([04a6ef1](https://github.com/KalvadTech/lecode/commit/04a6ef127ca1a99a6324edf664ebb9bcb333e5e3))
* focus memory documentation on lecode ([5a09ab8](https://github.com/KalvadTech/lecode/commit/5a09ab8ef5c0f8b1c188c01036fd6acd6c268a78))
* state Python 3.13 minimum in AGENTS.md ([d2129dd](https://github.com/KalvadTech/lecode/commit/d2129dd7d911eeaf89be808d49b663ef8e2aec25))

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
