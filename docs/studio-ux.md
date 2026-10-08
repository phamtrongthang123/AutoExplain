# AutoExplain local UX direction

## Reference inspected

The reference is convenience, not Unsloth branding, training features or an
assertion of equivalent model support. Reviewed official public documentation:

- [Install and first launch](https://unsloth.ai/docs/new/studio/install)
- [Model selection and task progress](https://unsloth.ai/docs/new/studio/start)
- [Chat, local models and comparisons](https://unsloth.ai/docs/new/studio/chat)

Also visually inspected three screenshots embedded in the official start guide:
the account-setup card, the chat landing screen with example prompts and composer,
and the studio overview with grouped model/dataset/parameter/progress panels.
This was public documentation/screenshot review, not a live Unsloth session.
The screenshots show a clear primary action, examples, grouped controls and
secondary settings; documentation describes searchable model selection and
explicit preparation/ready states. Details differ between documented versions.
No third-party assets are copied into AutoExplain.

## AutoExplain's user journey

1. **Start with a question:** explain text, explain an image, or explore an
   activation in language. Offer a no-download demo and meaningful examples.
2. **Prepare a suitable model:** recommend supported models and hardware defaults;
   display downloaded versus loaded state separately. Keep local paths, revisions
   and dtype overrides in Advanced. Never fetch weights without explicit consent.
3. **Explain:** make the input and action central. Show readable, signed token
   highlights or image overlays before numerical tables. Always identify the
   output being explained; a next-token attribution is not a whole-answer proof.
4. **Change and compare:** show original and changed outputs side by side with
   the same target and a clear change summary. Preserve local run history and
   exports while labeling stale results honestly.
5. **Manage work:** display the current preparation/execution stage, progress
   where measurable, recovery actions and unload controls. Cancellation must
   match actual backend boundaries, not just remove a spinner.

## NLA and local hardware

NLA must be a connected workflow rather than a prerequisite-only page: obtain
or select its actual source/AV/AR checkpoints, select source tokens, verbalize,
reconstruct, edit a description and compare a controlled source-model outcome.
Hardware inspection for this development session found a 10 GiB RTX 3080 and
about 48 GiB available system RAM. Loading all three unquantized Qwen models
onto that GPU is not a viable default. Use explicitly staged model lifetimes
and conservative CPU memory estimates; disclose speed and fit uncertainty.
Do not silently quantize the published NLA checkpoints or claim that a small
unrelated model substitutes for them.

## Delivery standard

Keep AutoExplain's existing local Streamlit application and safety contracts.
No new frontend framework, account service, cloud inference or training product
is required. Preserve useful existing workflows and expose technical details
progressively. This document is a design basis, not evidence of implemented or
runtime-validated features. No application execution, tests or model inference
has been performed as part of this reference review.
