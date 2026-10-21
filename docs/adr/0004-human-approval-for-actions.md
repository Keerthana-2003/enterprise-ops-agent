# ADR-0004: Agents propose actions; humans approve them

**Status:** accepted

**Context.** Actions such as service credits, collections notices or credit holds affect money and
customer relationships, and model output is probabilistic.

**Decision.** Data tools are read-only (SQL restricted to single SELECT statements on read-only
connections). The only action tool records a proposal with status `pending_human_approval`.

**Consequences.** No irreversible action without a person. Slightly slower end-to-end process, which
is acceptable for these workflows. Automation of low-risk actions can be revisited per action type
once evaluation data shows consistently correct proposals.
