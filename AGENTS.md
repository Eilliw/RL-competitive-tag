# AGENTS.md

Guidelines for future coding agents:

1. Keep changes minimal and focused on requested behavior.
2. Preserve the `cat_mouse_rl` package/module boundaries.
3. Add or update tests for behavior you change.
4. Prefer CLI/configuration options over hardcoded experiment values.
5. Do not fabricate experiment results.
6. Use MPE2 imports (`mpe2.simple_tag_v3`), not deprecated `pettingzoo.mpe`.
