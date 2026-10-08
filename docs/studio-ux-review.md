# Studio: six sequential rendered UX cycles

Six sequential review → fix → rendered re-review cycles completed. This document distinguishes observed UI behavior from source-reviewed model execution paths. Chromium only, existing app at `http://127.0.0.1:8501/`, CDP on loopback port 9224. Desktop viewport: 1440 × 1000; final narrow saved-record review: 390 × 844. Every linked before/after screenshot was saved and read. No automated tests, installs, model downloads, new inference, training or commits were run.

## Cycle 1 — first-launch hierarchy (completed)

[Before](assets/studio-ux/cycle-01-before.png) · [After](assets/studio-ux/cycle-01-after.png)

- Observed: repeated prose, large success/warning banners, radio sidebar and runtime diagnostics pushed input/output cards down to about y=655.
- Changed: concise recorded badge, primary “Use this example” action, direct navigation buttons, contextual runtime/privacy expander, shorter top spacing. Scientific failure/control warnings remain attached to the relevant intervention rather than a blanket startup warning.
- Re-rendered and read: cards begin about y=497; exact prompts and next-token outputs are visible on desktop. Sidebar is substantially shorter. Numerical tables still occupy excessive space; reserved for comparison cycle.
- Operational limit: refreshing applied the main script but not imported page/theme modules. User explicitly authorized server-only restarts as needed; root restarted the existing foreground `.venv-studio` server in `wN:pE`, same port/environment, producing PID 2895553. Session reset; no inference was active. Browser remained open. After screenshot was recaptured and read after the restart.

## Cycle 2 — model readiness (completed)

[Before](assets/studio-ux/cycle-02-before.png) · [After](assets/studio-ux/cycle-02-after.png)

- Observed: model selection was buried in each workflow; large hardware prose preceded status and Prepare (below the viewport).
- Changed: unified Models navigation with supported SmolLM, CLIP, Gemma and activation-language entry points. Cards show recommended use, local-file status and fit estimates. Text/image preparation now shows explicit Not on device / On device / Ready states; hardware, dtype, revision and trusted local path are contextual diagnostics.
- Interacted: Models → Prepare text → Model setup. Confirmed SmolLM files are missing and Prepare is disabled without explicit download consent. CLIP is also shown as missing. No consent checkbox was enabled and no transfer or load was started.
- Re-rendered and read: common model cards, fit guidance and action buttons are visible; NLA card begins near the desktop fold. Gemma readiness remains explicitly unchecked until requested; no fabricated ready state. NLA retains its specialist staged preparation rather than pretending it shares small-model resource requirements.

## Cycle 3 — editable example journey (completed)

[Before](assets/studio-ux/cycle-03-before.png) · [After](assets/studio-ux/cycle-03-after.png)

- Observed: live Gemma view exposed interpreter/version plumbing and only hardcoded pairs, although the real adapter accepts aligned prompt pairs. Primary example action led to a different small-model workflow.
- Changed: Use this example now seeds the selected real pair into an in-page Edit & run view. Donor/recipient, target/foil and supported layer are editable. The existing isolated cache-only runner accepts bounded JSON through stdin, validates native 2–32-token lengths, equal alignment, exactly one nonfinal changed token and distinct single-token targets before loading weights. Preserves time/output caps, offline flags, process cleanup and interprocess exclusion.
- Root performed the second authorized server restart (PID 2911762); no inference active, unsaved session reset. Reloaded Chromium, clicked Use this example, saved/read the editable input view, edited the donor to Italy and confirmed the Run button remains disabled without explicit run consent.
- Manually clicked the cache/backend/resource readiness check (no weights loaded). It returned Not ready. An existing `.venv-isaac51/bin/python` GPU compute process (PID 2833553, 2396 MiB at inspection) was active, so no inference was started. Successful inference/output and native-token error paths are source-reviewed, not claimed runtime-verified.

## Cycle 4 — comparison hierarchy (completed)

[Before](assets/studio-ux/cycle-04-before.png) · [After](assets/studio-ux/cycle-04-after.png)

- Observed: donor-token and top-five tables occupied the entire first viewport; the actual before/after intervention comparison was below those tables.
- Changed: exact token/probability tables and random/self-control tables are collapsed, preserving values. Main comparison uses fixed-target margin wording (also correct for custom prompts); signed token highlights precede their collapsed numeric inspector. Removed the permanently France-specific banner, which was misleading for other selections. Scientific failures, trivial final-readout warnings, recovery ratios and complete provenance remain.
- Root performed third authorized server refresh (PID 2921881), with no requested inference active; session reset. Re-rendered and read the comparison area: exact Berlin → Paris, −7.125 → 6.75, +13.875 and 0.8987854251012146 recovery appear in panels, with collapsed controls. The after image is scrolled to the comparison (not a same-scroll height comparison). Italy failure remains in source/report; attempts to select it in this browser interaction did not complete, so no claim of visually verifying that selection.

## Cycle 5 — recovery and continuity (completed)

[Before](assets/studio-ux/cycle-05-before.png) · [After](assets/studio-ux/cycle-05-after.png)

- Observed by interaction: edit donor to Italy → Models → Recorded pipelines → Edit & run reset donor to France. Existing task feedback disappeared on navigation; previous live results were cleared before retry.
- Changed: session-owned task records track Running / Completed / Failed / Cancelled and meaningful stages, with actionable retry guidance. Native Stop still unwinds backend cleanup before cancellation is recorded. No worker threads/shared mutable models. Draft values are detached from Streamlit unrendered-widget cleanup; consent/upload values are not silently retained. Last successful Gemma/text/image results survive failed retries and draft changes with explicit stale-result labeling. NLA foreground actions use the same task state and retain existing release blocks. Known static input-validation messages now provide corrective detail without echoing arbitrary exceptions.
- Root performed fourth authorized server refresh (PID 2936943), no requested inference active; session reset. Repeated Italy edit → Models → Recorded pipelines and confirmed **Italy retained**. Manually ran readiness check; read after screenshot showing **Gemma readiness · FAILED**, elapsed 1.7 s and corrective retry guidance. Opened diagnostics: dependencies compatible; pinned snapshot missing/incomplete; available GPU 5.785 GiB below 6 GiB requirement, available host RAM 42.428 GiB. No weights loaded or download attempted. Native Stop during inference and completed live output remain unexercised.

## Cycle 6 — durable experiments (completed)

[Before](assets/studio-ux/cycle-06-before.png) · [After](assets/studio-ux/cycle-06-after.png)

- Observed: only session history/export existed, explicitly lost at restart; no Open/Delete local experiment workflow.
- Changed: explicit Save controls for Gemma, text/image and NLA results; Saved experiments navigation with Open, JSON export and confirmation-gated selected Delete. Fixed app-owned SQLite store under XDG data/home local data; UUID identifiers and parameterized queries, bounded JSON depth/size/count/total, no pickle/arbitrary execution. Save is the only creation path; names never become paths. Privacy notices explain server-local unencrypted/shared storage and exclusion of checkpoint paths, credentials and original image bytes. Unsaved use remains transient.
- Root performed fifth authorized refresh (PID 2955996); no inference active, session reset. Viewed the empty Saved experiments page, then explicitly saved recorded Gemma as **UX review · recorded Gemma** and opened it. Read desktop screenshot showing privacy notice, Open, disabled-until-confirmed Delete and the real recorded result. Created only review record `cfd7cbcae2cb457dbfc3e11028b89bec` (14,956 JSON characters); no user-created records existed in the displayed store.
- Reviewed the saved view at 390 × 844: sidebar collapses, controls stack, heading and privacy notice wrap; measured document width remains 390 (no whole-page horizontal overflow). Long experiment selector text truncates visually but opened title displays below. The after screenshot was temporarily recaptured/read narrow; final desktop capture will replace it after persistence review. No app-wide mobile coverage claimed.
- Source review found saved selector choices were recorded but not restored; added restoration of bounded pair/patch defaults on Open before the final restart. Updated local-app guide and README to describe current behavior and distinguish rendered review from unexecuted model work.
- Root performed sixth authorized restart (PID 2967112), with no requested inference active. Restored desktop width, reopened **UX review · recorded Gemma**, and confirmed real input/output/margins still render after session reset. Recaptured/read final desktop after image. Verified Delete is initially disabled, clicked its explicit confirmation and deleted only review record `cfd7cbcae2cb457dbfc3e11028b89bec`. Empty-state UI and read-only SQLite count both confirmed zero remaining records. The empty app-owned database remains; no caches, original reports or unrelated data were deleted.

## Final scope and remaining limits

- All six cycles used the actual running Chromium UI sequentially, not six commits or source-only pseudo-reviews. No children were spawned. Root performed six narrowly authorized server-only restarts because imported modules stayed stale; each reset unsaved state with no requested inference active. Browsers stayed open.
- Manual evidence covers first-run hierarchy, model/no-download states, example input editing, comparison presentation, navigation retention, failed readiness recovery, explicit saving, restart persistence, opening and selected deletion. Original/bundled Gemma hashes were rechecked after completion and remain identical.
- Exactly two manual Gemma readiness checks ran, with no weight loading. Latest observed diagnostic: compatible split backend, pinned cache incomplete/missing, 5.785 GiB GPU free versus 6 GiB required. An unrelated GPU compute job was left untouched. No SmolLM/CLIP/NLA/Gemma inference or model download ran.
- Successful live inference, native-token rejection after ready preflight, running/completed/cancelled inference states and graceful Stop resource cleanup were source-reviewed, **not runtime-demonstrated**. The displayed failed readiness state was genuine, not injected. No automated suite or test-only build was run.
- Narrow review covered Saved experiments only: controls stack without whole-page overflow. Long selector labels truncate; opened titles wrap. Other mobile workflows, screen readers and all model-loaded states are not claimed reviewed. Red Streamlit selection accents remain in some native controls; no framework/theme rewrite was attempted.
- Saved image records retain hash/scores/attributions, not original pixels or a restorable overlay. NLA reopening presents its explicit input/language/numerical provenance rather than restoring live tensors. Local storage is unencrypted and shared by the app OS account; no app authentication project was added.
- Final browser handoff: existing Chromium app tab returned to Recorded pipelines at desktop dimensions; saved store empty after review cleanup. No further background work remains.

## Protected evidence

Original and bundled Gemma reports have not been edited. Initial matching SHA-256 values:

- `gemma4_pretrained.json`: `ed109c66ff1fbf27cd93196dcea059412554f24946e07d02fd49799c45d93dff`
- `gemma4_jlens.json`: `a09fcb916c19b02cfe437f802fcd199acc756aeb3c9a966b5b62de83288c681b`
