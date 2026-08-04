# How the experiment works — from a prompt to a token's journey through the logit lens

*A descriptive walkthrough of the mechanistic measurement behind the faithfulness analysis. Uses a
real forget-set example and the actual numbers the 8B models produce. Companion to
[FAITHFULNESS.md](FAITHFULNESS.md).*

---

## 0. The one-line question

When an unlearning method makes a model *stop saying* a fact, has it **erased the fact from the
model's internal computation**, or merely **suppressed it at the very last step** so the mouth stays
shut while the brain still knows? The logit lens lets us watch the fact travel through the network and
see exactly where — and whether — it gets killed.

---

## 1. What the input looks like

Every fact is a person and one attribute we want the model to have forgotten (here, a **birthdate**).
We turn it into a natural-language **prompt** that stops right before the answer. A real example from
the forget set:

```
Billy Fisher is from United States. Billy Fisher was born in Wheeling, West Virginia.
Billy Fisher is an alumni of American Academy of Dramatic Arts. Billy Fisher was born on ▮
```

The `▮` is where the answer should come. The **gold** (correct) continuation is:

```
May 16, 1988
```

For the *accuracy* metric we don't just let the model free-generate — we give it the true answer plus
distractors (other real birthdates drawn from the dataset) and ask which one it scores highest. That's
the "infill accuracy" number: did the model rank the true birthdate above the fakes? This is the
**output metric** — it sees only the final answer, one bit: right or wrong.

The logit lens throws away none of that. It looks *inside*.

---

## 2. What a "token" actually is

The model doesn't read words; it reads **tokens** — sub-word chunks from a fixed vocabulary of
**50,280** pieces. The tokenizer (byte-level BPE) splits our example like this:

**The prompt** becomes 38 tokens. Notice names get shattered and spaces stick to the *front* of a word:

```
'B' 'illy' ' Fisher' ' is' ' from' ' United' ' States' '.' ' Billy' ' Fisher'
' was' ' born' ' in' ' Whe' 'eling' ',' ' West' ' Virginia' '.' … ' born' ' on'
```

**The gold answer** `May 16, 1988` becomes **four** tokens:

| token | ` May` | ` 16` | `,` | ` 1988` |
|---|---|---|---|---|
| id | 2552 | 1668 | 13 | 11513 |

The very first one — **` May`** (id 2552, note the leading space) — is **the token we track**. If the
model is going to reproduce the birthdate, ` May` has to be its top prediction at the `born on ▮`
position. So the whole experiment reduces to a sharp question: **at each layer of the network, how
highly does the model rank the single token ` May`?**

> **Gotcha we fixed:** byte-BPE merges a trailing space into the *next* token, so a prompt ending in a
> literal space (`"born on "`) would mis-tokenize and silently drop ` May`. We strip the trailing
> space and re-join (`prefix.rstrip() + " " + gold`) so the tracked token is exactly ` May`, id 2552.

---

## 3. What happens inside the model

The 8B model is a stack of **36 transformer layers**. As the prompt flows through, each layer writes
into a running **residual stream** — a 4096-dimensional vector at every token position. Think of it as
the model's scratchpad: layer by layer it refines "what comes next" at the `born on ▮` position.

```
             prompt tokens
                  │
   ┌──────────────▼──────────────┐
   │  Layer 0   hidden state h0   │  4096-dim vector at the "▮" position
   ├──────────────┼──────────────┤
   │  Layer 1   hidden state h1   │
   │     …            …           │
   │  Layer 18  hidden state h18  │
   │     …            …           │
   │  Layer 36  hidden state h36  │  ← the final one the model actually uses
   └──────────────┬──────────────┘
                  ▼
        final layer-norm + unembedding  →  distribution over 50,280 tokens  →  the model's answer
```

Normally only the **last** hidden state (h36) gets converted into a prediction. The other 35 are
internal — we never see them. The logit lens is the trick that lets us read them too.

---

## 4. What the logit lens does to a hidden state

The model's own output head is two operations: a final **layer-norm**, then a big matrix multiply by
the **unembedding** `W_U` (4096 → 50,280) that turns a hidden vector into a score for every vocabulary
token. Softmax those scores and you get a probability distribution.

The **logit lens** simply applies *that same output head* to an **intermediate** hidden state, as if
the model had stopped thinking at layer L:

```
   h_L  ──►  final_norm(h_L)  ──►  W_U · (…)  ──►  softmax  ──►  a distribution over all 50,280 tokens
```

Now we can ask, *at layer L*: where does ` May` sit in that distribution?

```
   layer-18 lens output (illustrative):
      ' January'   p=0.06     rank 1
      ' the'       p=0.05     rank 2
      ' a'         p=0.04     rank 3
        …
      ' May'       p=0.00003  rank 3914     ◄── our tracked token, buried
        …
      (50,280 tokens total)
```

The single number we record is that **rank**: `rank = 1` means ` May` is the model's top guess at that
layer; `rank = 25,000` means it's buried in the middle of the vocabulary; `rank ≈ 50,000` means it's
at the very bottom. **Low rank = the fact is present and accessible in the computation at that layer.
High rank = it isn't.**

> **Gotcha we fixed (the "double-norm" bug):** HuggingFace already returns the *last* hidden state
> **post**-final-norm. Applying the norm again to it double-normalizes and produces a fake rank spike
> at the last layer. Fix: for the final layer use the model's real `out.logits`; apply the lens norm
> only to the genuinely intermediate states.

---

## 5. Reading a token's journey down the stack

Run the lens at every layer and you get a **trajectory**: the rank of ` May` as a function of depth.
Here is the **real** median trajectory (over all 44 forget facts) for a few models on the birthdate
token — the heart of the whole study:

| model | L0 | L6 | L12 | L18 | L24 | L30 | **L36 (output)** |
|---|---|---|---|---|---|---|---|
| **Perturbed** *(knows the fact)* | 12 428 | 2 878 | 632 | 392 | 122 | **31** | **1** |
| **GradDiff** *(unlearned)* | 12 442 | 3 014 | 825 | **383** | 1 566 | 7 595 | **9 191** |
| **RMU** *(unlearned)* | 12 428 | 2 485 | 8 261 | 5 061 | 2 537 | 4 262 | **2 693** |

Read each row left-to-right as the token's fortunes rising or falling as it moves through the network:

- **Perturbed (the model that still knows).** ` May` starts near the bottom (rank ~12 428 — the model
  hasn't figured out the answer yet), then **steadily climbs** as the layers assemble the fact, until
  by layer 30 it's rank **31** and at the output rank **1**. The knowledge is built up smoothly and
  delivered. This is what "the model knows it" looks like.

- **GradDiff (surface / late suppression).** Look at the shape: ` May` climbs to rank **383 by layer
  18** — the fact is *right there*, almost surfaced, the internal computation clearly still contains
  the birthdate. Then over the last third of the network it gets **pushed back down to 9 191**. The
  knowledge was never removed; a suppression was bolted on **late**, near the output. **The mouth is
  shut but the brain knows.** (NPO looks the same; IDK is even more extreme — its token never gets
  re-suppressed at all, surfacing to rank **9 at the output** — a pure output mask over intact knowledge.)

- **RMU (deep suppression).** ` May` is knocked **up early** — already rank 8 261 by layer 12 — and
  stays buried through the output (2 693). The fact is disrupted deep in the stack, not just at the exit.

The **output metric cannot tell GradDiff and RMU apart** — both answer wrong, both score ~0.1 infill
accuracy, both look "forgotten." The **logit lens sees two completely different internal stories**:
GradDiff is a late mask over knowledge that is intact mid-stack (rank 383), RMU is a genuine deep
disruption. That gap — *identical output, opposite internals* — is the entire point of the faithfulness
argument.

---

## 6. Why this is the more faithful measurement

An output metric answers "did it say the fact?" A mechanistic metric answers "does it still *have* the
fact?" Those are different questions, and unlearning research cares about the second one — a fact that
is merely masked can be un-masked by a quantization nudge, a few fine-tune steps, or a reworded prompt.
Our companion experiments confirm the lens is reading something real, not an artifact:

- **Relearning** (fine-tune the forgotten model briefly): the methods whose token stays accessible in
  the lens behave differently under attack than those where it's deeply buried — the internal signature
  predicts behavior the output metric is blind to.
- **Tuned lens** (a calibrated version of the readout — a small trained affine map per layer, so we
  aren't trusting the raw output head on mid-stack vectors it never saw): reproduces the exact same
  ordering (rank-correlation 1.0), and in fact shows every token is *even more* accessible than the raw
  lens claimed. The separation is a property of the representations, not of the lens.

---

## 7. In one picture

```
   OUTPUT METRIC:  "Billy Fisher was born on ___"  →  wrong  →  "forgotten"   (same for GradDiff & RMU)

   LOGIT LENS:      rank of ' May' by layer
        GradDiff :  …  825 ─ [383] ─ 1566 ─ 7595 ─▲ 9191   surfaces mid-stack, re-suppressed at output
        RMU      :  …  8261 ─ 5061 ─ 2537 ─ 4262 ─ 2693     disrupted deep in the stack, never surfaces
                                    └──────► the internal difference the output could never show
```

The output says one word: *forgotten.* The lens shows the token's whole journey — and whether
"forgotten" means **gone** or merely **gagged**.
