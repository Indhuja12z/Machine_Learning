import numpy as np
import gradio as gr

# ── States & Observations ──────────────────────────────────────────────────
states       = ["Rainy", "Sunny"]
observations = ["Walk", "Shop", "Clean"]
n_states = len(states)
n_obs    = len(observations)

# Default model parameters (never mutated globally)
DEFAULT_START = np.array([0.5, 0.5])
DEFAULT_TRANS = np.array([[0.6, 0.4],
                           [0.3, 0.7]])
DEFAULT_EMIT  = np.array([[0.1, 0.4, 0.5],
                           [0.6, 0.3, 0.1]])

# Mutable working copies (only learning updates these)
start_p = DEFAULT_START.copy()
trans_p = DEFAULT_TRANS.copy()
emit_p  = DEFAULT_EMIT.copy()

# ── BUG 1 & 2 FIX: Robust input encoder ───────────────────────────────────
def encode_input(user_input):
    """
    FIX 1: Reject empty input with a clear message instead of crashing.
    FIX 2: Case-insensitive matching — 'walk' and 'Walk' both work.
    """
    stripped = user_input.strip()
    if not stripped:
        return "Error: Please enter at least one observation.", None

    tokens = stripped.split()
    indices = []
    invalid = []

    for token in tokens:
        # FIX 2: capitalise first letter for case-insensitive match
        normalised = token.capitalize()
        if normalised in observations:
            indices.append(observations.index(normalised))
        else:
            invalid.append(token)

    if invalid:
        return (f"Error: Unknown observation(s): {invalid}. "
                f"Valid values are: {observations}"), None

    encoded_str = " → ".join(f"{t.capitalize()}({i})"
                              for t, i in zip(tokens, indices))
    return f"Encoded: [{', '.join(map(str, indices))}]  |  {encoded_str}", indices


# ── Forward Algorithm (BUG 6 FIX: show full alpha matrix) ─────────────────
def run_forward(user_input):
    """
    FIX 6: Returns the full alpha matrix step-by-step, not just the final prob.
    """
    encoded_text, obs_seq = encode_input(user_input)
    if obs_seq is None:
        return encoded_text, ""

    T = len(obs_seq)
    alpha = np.zeros((T, n_states))
    alpha[0] = start_p * emit_p[:, obs_seq[0]]

    for t in range(1, T):
        for j in range(n_states):
            alpha[t, j] = (np.sum(alpha[t - 1] * trans_p[:, j])
                           * emit_p[j, obs_seq[t]])

    prob = np.sum(alpha[T - 1])

    # Build detailed output
    lines = ["═" * 45]
    lines.append("  FORWARD ALGORITHM — α (alpha) matrix")
    lines.append("═" * 45)
    for t in range(T):
        obs_name = observations[obs_seq[t]]
        lines.append(f"\n  t={t+1}  Observe: {obs_name}")
        lines.append(f"  {'State':<10} {'α value':>12}")
        lines.append(f"  {'-'*24}")
        for s in range(n_states):
            lines.append(f"  {states[s]:<10} {alpha[t, s]:>12.6f}")

    lines.append("\n" + "═" * 45)
    lines.append(f"  P(O | λ)  =  {prob:.6f}")
    lines.append(f"            ≈  {prob * 100:.4f}%")
    lines.append("═" * 45)

    return encoded_text, "\n".join(lines)


# ── Viterbi Algorithm (BUG 7 FIX: show delta matrix + path prob) ──────────
def run_viterbi(user_input):
    """
    FIX 7: Returns the full delta matrix, backpointers, and best path prob.
    """
    encoded_text, obs_seq = encode_input(user_input)
    if obs_seq is None:
        return encoded_text, ""

    T = len(obs_seq)
    delta   = np.zeros((T, n_states))
    psi     = np.zeros((T, n_states), dtype=int)

    delta[0] = start_p * emit_p[:, obs_seq[0]]

    for t in range(1, T):
        for j in range(n_states):
            scores     = delta[t - 1] * trans_p[:, j]
            psi[t, j]  = np.argmax(scores)
            delta[t, j] = np.max(scores) * emit_p[j, obs_seq[t]]

    # Traceback
    best_path       = [0] * T
    best_path[T - 1] = int(np.argmax(delta[T - 1]))
    for t in range(T - 2, -1, -1):
        best_path[t] = psi[t + 1, best_path[t + 1]]

    best_prob    = delta[T - 1, best_path[T - 1]]
    result_states = [states[i] for i in best_path]

    # Build detailed output
    lines = ["═" * 45]
    lines.append("  VITERBI ALGORITHM — δ (delta) matrix")
    lines.append("═" * 45)
    for t in range(T):
        obs_name = observations[obs_seq[t]]
        lines.append(f"\n  t={t+1}  Observe: {obs_name}")
        lines.append(f"  {'State':<10} {'δ value':>12}  {'←from':>8}")
        lines.append(f"  {'-'*34}")
        for s in range(n_states):
            prev = states[psi[t, s]] if t > 0 else "—"
            lines.append(f"  {states[s]:<10} {delta[t, s]:>12.6f}  {prev:>8}")

    lines.append("\n" + "═" * 45)
    lines.append(f"  Best path:  {' → '.join(result_states)}")
    lines.append(f"  P(Q*|O,λ)  =  {best_prob:.6f}")
    lines.append("═" * 45)

    return encoded_text, "\n".join(lines)


# ── Learning (BUG 3 & 4 & 5 FIX) ──────────────────────────────────────────
def run_learning(user_input):
    """
    FIX 3: Parses user input as training pairs instead of hardcoded data.
    FIX 4: Uses local variables; only updates globals after full computation.
    FIX 5: Adds Laplace smoothing to prevent NaN when a state is never seen.
    
    Input format: space-separated state:obs pairs
    Example:  Rainy:Clean Rainy:Shop Sunny:Walk Sunny:Walk Rainy:Clean
    """
    global start_p, trans_p, emit_p

    stripped = user_input.strip()
    if not stripped:
        return ("Error: Enter training pairs like:  "
                "Rainy:Clean Sunny:Walk Rainy:Shop"), ""

    tokens = stripped.split()
    training_data = []

    for token in tokens:
        parts = token.split(":")
        if len(parts) != 2:
            return (f"Error: Bad format '{token}'. "
                    f"Use State:Obs pairs, e.g. Rainy:Clean"), ""
        s, o = parts[0].capitalize(), parts[1].capitalize()
        if s not in states:
            return f"Error: Unknown state '{s}'. Valid: {states}", ""
        if o not in observations:
            return f"Error: Unknown obs '{o}'. Valid: {observations}", ""
        training_data.append((s, o))

    # Count transitions and emissions
    # FIX 5: Laplace smoothing (+1) prevents zero rows → no NaN
    trans_counts = np.ones((n_states, n_states))   # smoothed
    emit_counts  = np.ones((n_states, n_obs))       # smoothed
    start_counts = np.zeros(n_states)

    for i, (s, o) in enumerate(training_data):
        s_idx = states.index(s)
        o_idx = observations.index(o)
        emit_counts[s_idx, o_idx] += 1
        if i == 0:
            start_counts[s_idx] += 1
        if i < len(training_data) - 1:
            next_s = training_data[i + 1][0]
            trans_counts[s_idx, states.index(next_s)] += 1

    # FIX 4: Compute locally, assign to globals only if all valid
    new_start = start_counts / np.sum(start_counts) if np.sum(start_counts) > 0 else DEFAULT_START.copy()
    new_trans = trans_counts / trans_counts.sum(axis=1, keepdims=True)
    new_emit  = emit_counts  / emit_counts.sum(axis=1, keepdims=True)

    start_p = new_start
    trans_p = new_trans
    emit_p  = new_emit

    lines = ["═" * 45]
    lines.append("  MODEL TRAINED SUCCESSFULLY")
    lines.append(f"  Sequences used: {len(training_data)}")
    lines.append("═" * 45)
    lines.append("\n  Start Probabilities π:")
    for s, p in zip(states, start_p):
        lines.append(f"    {s:<10}  {p:.4f}")
    lines.append("\n  Transition Matrix A:")
    lines.append(f"    {'':10}" + "".join(f"{s:>10}" for s in states))
    for i, row in enumerate(trans_p):
        lines.append(f"    {states[i]:<10}" + "".join(f"{v:>10.4f}" for v in row))
    lines.append("\n  Emission Matrix B:")
    lines.append(f"    {'':10}" + "".join(f"{o:>8}" for o in observations))
    for i, row in enumerate(emit_p):
        lines.append(f"    {states[i]:<10}" + "".join(f"{v:>8.4f}" for v in row))
    lines.append("\n  (Forward/Viterbi now use these updated params)")
    lines.append("═" * 45)

    return f"Trained on {len(training_data)} pairs", "\n".join(lines)


# ── Gradio UI ──────────────────────────────────────────────────────────────
with gr.Blocks(title="HMM Explorer") as app:
    input_box      = gr.Textbox(label="Input", placeholder="Walk Shop Clean")
    encoded_output = gr.Textbox(label="Encoded Input")
    result_output  = gr.Textbox(label="Result", lines=18)

    with gr.Row():
        btn_forward  = gr.Button("▶ Forward (Evaluation)",  variant="primary")
        btn_viterbi  = gr.Button("▶ Viterbi (Decoding)",    variant="primary")
        btn_learning = gr.Button("▶ Learning (Baum-Welch)", variant="secondary")

    btn_forward.click(run_forward,  inputs=input_box,
                      outputs=[encoded_output, result_output])
    btn_viterbi.click(run_viterbi,  inputs=input_box,
                      outputs=[encoded_output, result_output])
    btn_learning.click(run_learning, inputs=input_box,
                       outputs=[encoded_output, result_output])

app.launch()
