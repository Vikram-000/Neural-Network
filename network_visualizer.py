"""
Real-Time Neural Network Decision Display for CartPole-v1.

Visualizes the architecture, neuron activations, connection weights,
live observation inputs, and output decisions of a trained NeuralNetwork
in real time alongside the CartPole-v1 simulation.
"""

import math
import sys
import gymnasium as gym
import numpy as np
import pygame
from basic_neural_network import NeuralNetwork

# Typography helper
def get_font(size: int, bold: bool = False) -> pygame.font.Font:
    """Return an available system font or default font with fallback."""
    if not pygame.font.get_init():
        pygame.font.init()
    for name in ["Segoe UI", "Arial", "Calibri", "DejaVu Sans", "Helvetica"]:
        try:
            matched = pygame.font.match_font(name, bold=bold)
            if matched:
                return pygame.font.Font(matched, size)
        except Exception:
            pass
    return pygame.font.Font(None, size)


class NetworkVisualizer:
    """Independent renderer for Neural Network decision-making and telemetry."""

    # Color Palette (Modern Dark Theme)
    BG_COLOR = (15, 18, 25)
    PANEL_BG = (22, 27, 38)
    CARD_BG = (29, 36, 50)
    BORDER_COLOR = (44, 55, 77)
    TEXT_MAIN = (240, 243, 248)
    TEXT_MUTED = (140, 153, 173)
    TEXT_ACCENT = (79, 172, 254)

    # Positive / Negative weights
    POS_WEIGHT_COLOR = (52, 152, 219)   # Soft Cyan-Blue
    NEG_WEIGHT_COLOR = (243, 114, 114)  # Soft Coral-Red
    ACTIVE_FLOW_COLOR = (0, 245, 160)   # Highlighting active excitatory paths

    # Node Activation Colors
    NODE_INACTIVE = (45, 55, 72)
    NODE_ACTIVE = (46, 213, 115)        # Emerald glow for ReLU firing
    OUTPUT_LEFT = (0, 180, 255)         # Cyan
    OUTPUT_RIGHT = (255, 159, 26)       # Amber-Gold
    HALO_COLOR = (255, 220, 90)         # Selected action glow

    # Observation Feature Labels
    FEATURE_LABELS = [
        {"name": "Cart Position", "sym": "x", "unit": "m", "min": -2.4, "max": 2.4},
        {"name": "Cart Velocity", "sym": "v", "unit": "m/s", "min": -3.0, "max": 3.0},
        {"name": "Pole Angle", "sym": "θ", "unit": "deg", "min": -15.0, "max": 15.0},
        {"name": "Pole Angular Vel", "sym": "ω", "unit": "deg/s", "min": -180.0, "max": 180.0},
    ]

    ACTION_LABELS = ["◄ PUSH LEFT", "PUSH RIGHT ►"]

    def __init__(self, rect: pygame.Rect):
        self.rect = rect
        self.font_title = get_font(18, bold=True)
        self.font_subtitle = get_font(14, bold=True)
        self.font_body = get_font(13, bold=False)
        self.font_small = get_font(11, bold=False)
        self.font_code = get_font(12, bold=False)

        # Cache for node layout positions
        self.node_positions = {}
        self.last_hovered_node = None

    def _compute_layout(self, network: NeuralNetwork, graph_rect: pygame.Rect):
        """Compute (x, y) coordinates for all neuron nodes dynamically."""
        left_x = graph_rect.left + 95
        right_x = graph_rect.right - 135
        hidden_x = (left_x + right_x) // 2

        layers = [
            ("input", network.input_size, left_x),
            ("hidden", network.hidden_size, hidden_x),
            ("output", network.output_size, right_x),
        ]

        positions = {"input": [], "hidden": [], "output": []}

        for layer_name, count, x in layers:
            total_height = graph_rect.height - 40
            node_spacing = total_height / max(1, count - 1) if count > 1 else 0

            for i in range(count):
                if count == 1:
                    y = graph_rect.centery
                else:
                    y = graph_rect.top + 20 + i * node_spacing
                positions[layer_name].append((int(x), int(y)))

        return positions

    def draw(
        self,
        surface: pygame.Surface,
        network: NeuralNetwork,
        observation: np.ndarray,
        action: int,
        step: int,
        episode: int,
        mouse_pos=None,
    ):
        """Render the complete neural network decision display panel."""
        # Draw panel background
        pygame.draw.rect(surface, self.PANEL_BG, self.rect)
        pygame.draw.rect(surface, self.BORDER_COLOR, self.rect, width=1)

        # 1. Header Banner: Title & Live Decision Pill
        self._draw_header(surface, network, action, step, episode)

        # 2. Probability Bar & Observation Breakdown
        self._draw_telemetry_bar(surface, network, observation, action)

        # 3. Main Neural Network Architecture Graph
        graph_rect = pygame.Rect(
            self.rect.left + 20,
            self.rect.top + 152,
            self.rect.width - 40,
            self.rect.height - 215,
        )
        self.node_positions = self._compute_layout(network, graph_rect)
        self._draw_network_graph(surface, network, observation, action, mouse_pos)

        # 4. Footer: Legend & Status Indicators
        self._draw_footer(surface)

        # 5. Tooltip on mouse hover (if applicable)
        if mouse_pos:
            self._draw_hover_tooltip(surface, network, observation, mouse_pos)

    def _draw_header(
        self,
        surface: pygame.Surface,
        network: NeuralNetwork,
        action: int,
        step: int,
        episode: int,
    ):
        header_y = self.rect.top + 12

        # Title
        title_surf = self.font_title.render("NEURAL NETWORK DECISION DISPLAY", True, self.TEXT_MAIN)
        surface.blit(title_surf, (self.rect.left + 20, header_y))

        # Topology badge
        topo_str = f"Arch: {network.input_size} In → {network.hidden_size} Hidden → {network.output_size} Out"
        topo_surf = self.font_small.render(topo_str, True, self.TEXT_MUTED)
        surface.blit(topo_surf, (self.rect.left + 20, header_y + 24))

        # Episode & Step counter
        stats_str = f"Episode: {episode}  |  Step: {step}/500"
        stats_surf = self.font_subtitle.render(stats_str, True, self.TEXT_ACCENT)
        stats_rect = stats_surf.get_rect(right=self.rect.right - 20, top=header_y + 4)
        surface.blit(stats_surf, stats_rect)

    def _draw_telemetry_bar(
        self,
        surface: pygame.Surface,
        network: NeuralNetwork,
        observation: np.ndarray,
        action: int,
    ):
        """Draw selected action pill and probability progress bar."""
        card_rect = pygame.Rect(
            self.rect.left + 20,
            self.rect.top + 55,
            self.rect.width - 40,
            64,
        )
        pygame.draw.rect(surface, self.CARD_BG, card_rect, border_radius=6)
        pygame.draw.rect(surface, self.BORDER_COLOR, card_rect, width=1, border_radius=6)

        # Probabilities from network output A2
        probs = network.A2[0] if hasattr(network, "A2") and network.A2 is not None else np.array([0.5, 0.5])
        p_left = float(probs[0])
        p_right = float(probs[1])

        # Action pill on the left
        pill_rect = pygame.Rect(card_rect.left + 12, card_rect.top + 12, 170, 40)
        pill_color = (18, 55, 40) if action == 0 else (55, 35, 18)
        border_col = self.NODE_ACTIVE if action == 0 else self.OUTPUT_RIGHT
        pygame.draw.rect(surface, pill_color, pill_rect, border_radius=5)
        pygame.draw.rect(surface, border_col, pill_rect, width=2, border_radius=5)

        act_text = self.ACTION_LABELS[action]
        act_surf = self.font_subtitle.render(act_text, True, (255, 255, 255))
        act_rect = act_surf.get_rect(center=pill_rect.center)
        surface.blit(act_surf, act_rect)

        # Probabilities Bar on the right
        bar_left = pill_rect.right + 25
        bar_width = card_rect.right - bar_left - 15
        bar_y = card_rect.top + 28
        bar_h = 16

        # Draw background bar
        pygame.draw.rect(surface, (20, 24, 34), (bar_left, bar_y, bar_width, bar_h), border_radius=4)

        # Split bar fill
        left_w = int(bar_width * p_left)
        if left_w > 0:
            pygame.draw.rect(surface, self.OUTPUT_LEFT, (bar_left, bar_y, left_w, bar_h), border_top_left_radius=4, border_bottom_left_radius=4)
        if bar_width - left_w > 0:
            pygame.draw.rect(surface, self.OUTPUT_RIGHT, (bar_left + left_w, bar_y, bar_width - left_w, bar_h), border_top_right_radius=4, border_bottom_right_radius=4)

        # Probability percentage labels
        lbl_left = self.font_small.render(f"P(Left): {p_left * 100:4.1f}%", True, self.OUTPUT_LEFT)
        lbl_right = self.font_small.render(f"P(Right): {p_right * 100:4.1f}%", True, self.OUTPUT_RIGHT)
        surface.blit(lbl_left, (bar_left, bar_y - 17))
        right_rect = lbl_right.get_rect(right=bar_left + bar_width, bottom=bar_y - 1)
        surface.blit(lbl_right, right_rect)

    def _draw_network_graph(
        self,
        surface: pygame.Surface,
        network: NeuralNetwork,
        observation: np.ndarray,
        action: int,
        mouse_pos,
    ):
        """Draw connections and neuron nodes."""
        inp_pos = self.node_positions["input"]
        hid_pos = self.node_positions["hidden"]
        out_pos = self.node_positions["output"]

        # Hidden activations A1
        hidden_acts = (
            network.A1[0]
            if hasattr(network, "A1") and network.A1 is not None
            else np.zeros(network.hidden_size)
        )
        output_acts = (
            network.A2[0]
            if hasattr(network, "A2") and network.A2 is not None
            else np.zeros(network.output_size)
        )

        # Max weight for line width scaling
        max_w1 = max(0.01, float(np.max(np.abs(network.W1))))
        max_w2 = max(0.01, float(np.max(np.abs(network.W2))))

        # ----------------- Connections: Input -> Hidden -----------------
        for i, p_in in enumerate(inp_pos):
            for h, p_hid in enumerate(hid_pos):
                w = float(network.W1[i, h])
                weight_norm = abs(w) / max_w1
                width = max(1, int(weight_norm * 3))
                alpha = int(30 + weight_norm * 140)

                color = self.POS_WEIGHT_COLOR if w >= 0 else self.NEG_WEIGHT_COLOR
                self._draw_translucent_line(surface, p_in, p_hid, color, alpha, width)

        # ----------------- Connections: Hidden -> Output -----------------
        for h, p_hid in enumerate(hid_pos):
            act_h = float(hidden_acts[h])
            is_firing = act_h > 0

            for o, p_out in enumerate(out_pos):
                w = float(network.W2[h, o])
                weight_norm = abs(w) / max_w2
                width = max(1, int(weight_norm * 3.5))

                # Highlight connection if hidden node is firing and feeds into selected action
                if is_firing and o == action and w > 0:
                    color = self.ACTIVE_FLOW_COLOR
                    alpha = min(255, int(120 + weight_norm * 135))
                    width = max(2, width + 1)
                else:
                    color = self.POS_WEIGHT_COLOR if w >= 0 else self.NEG_WEIGHT_COLOR
                    alpha = int(25 + weight_norm * 110)

                self._draw_translucent_line(surface, p_hid, p_out, color, alpha, width)

        # ----------------- Layer Headers -----------------
        for layer_name, title, col in [
            ("input", f"INPUTS ({network.input_size})", inp_pos[0][0]),
            ("hidden", f"HIDDEN RELU ({network.hidden_size})", hid_pos[0][0]),
            ("output", f"OUTPUTS ({network.output_size})", out_pos[0][0]),
        ]:
            header_surf = self.font_small.render(title, True, self.TEXT_MUTED)
            surface.blit(header_surf, (col - header_surf.get_width() // 2, self.rect.top + 130))

        # ----------------- Input Nodes -----------------
        obs_raw = observation.flatten() if observation is not None else np.zeros(network.input_size)
        for i, (x, y) in enumerate(inp_pos):
            val = float(obs_raw[i])
            meta = self.FEATURE_LABELS[i] if i < len(self.FEATURE_LABELS) else {"sym": f"x{i}", "unit": "", "min": -1, "max": 1}

            # Magnitude color
            norm = np.clip((val - meta["min"]) / (meta["max"] - meta["min"]), 0.0, 1.0)
            node_color = (
                int(40 + norm * 180),
                int(80 + (1 - abs(norm - 0.5) * 2) * 120),
                int(220 - norm * 100),
            )

            # Node circle
            pygame.draw.circle(surface, node_color, (x, y), 10)
            pygame.draw.circle(surface, self.BORDER_COLOR, (x, y), 10, width=2)

            # Value label next to input node
            if meta["unit"] == "deg":
                val_display = math.degrees(val)
                unit_str = "°"
            elif meta["unit"] == "deg/s":
                val_display = math.degrees(val)
                unit_str = "°/s"
            else:
                val_display = val
                unit_str = meta["unit"]

            lbl_str = f"{meta['sym']}: {val_display:+.2f}{unit_str}"
            lbl_surf = self.font_code.render(lbl_str, True, self.TEXT_MAIN)
            surface.blit(lbl_surf, (x - lbl_surf.get_width() - 14, y - 7))

        # ----------------- Hidden Nodes -----------------
        max_act = max(0.01, float(np.max(hidden_acts)))
        for h, (x, y) in enumerate(hid_pos):
            act_val = float(hidden_acts[h])
            is_active = act_val > 0

            if is_active:
                intensity = min(1.0, act_val / max_act)
                node_col = (
                    int(20 + 26 * intensity),
                    int(120 + 93 * intensity),
                    int(60 + 55 * intensity),
                )
                r = 8 + int(intensity * 3)
                # Outer glow for firing neuron
                self._draw_glow(surface, (x, y), r + 4, (46, 213, 115, 60))
            else:
                node_col = self.NODE_INACTIVE
                r = 7

            pygame.draw.circle(surface, node_col, (x, y), r)
            pygame.draw.circle(surface, self.BORDER_COLOR, (x, y), r, width=1)

        # ----------------- Output Nodes -----------------
        for o, (x, y) in enumerate(out_pos):
            prob = float(output_acts[o])
            is_selected = o == action
            base_color = self.OUTPUT_LEFT if o == 0 else self.OUTPUT_RIGHT

            # Node circle
            r = 13 if is_selected else 10
            pygame.draw.circle(surface, base_color, (x, y), r)

            # Selected action halo
            if is_selected:
                self._draw_glow(surface, (x, y), r + 8, (255, 220, 90, 110))
                pygame.draw.circle(surface, self.HALO_COLOR, (x, y), r + 3, width=2)

            # Output action label
            name_str = "PUSH LEFT" if o == 0 else "PUSH RIGHT"
            pct_str = f"{name_str} ({prob * 100:.1f}%)"
            name_surf = self.font_code.render(pct_str, True, base_color if is_selected else self.TEXT_MUTED)
            surface.blit(name_surf, (x + 18, y - 7))

    def _draw_footer(self, surface: pygame.Surface):
        """Draw bottom legend explaining symbols and colors."""
        foot_y = self.rect.bottom - 42
        foot_rect = pygame.Rect(self.rect.left + 20, foot_y, self.rect.width - 40, 32)
        pygame.draw.rect(surface, self.CARD_BG, foot_rect, border_radius=4)
        pygame.draw.rect(surface, self.BORDER_COLOR, foot_rect, width=1, border_radius=4)

        # Items in legend
        items = [
            ("line", self.POS_WEIGHT_COLOR, "(+) Weight"),
            ("line", self.NEG_WEIGHT_COLOR, "(-) Weight"),
            ("line", self.ACTIVE_FLOW_COLOR, "Active Flow"),
            ("dot", self.NODE_ACTIVE, "Firing ReLU"),
            ("dot", self.NODE_INACTIVE, "Zero ReLU"),
            ("halo", self.HALO_COLOR, "Selected Action"),
        ]

        curr_x = foot_rect.left + 14
        for kind, color, label in items:
            if kind == "line":
                pygame.draw.line(surface, color, (curr_x, foot_rect.centery), (curr_x + 16, foot_rect.centery), width=2)
                curr_x += 20
            elif kind == "dot":
                pygame.draw.circle(surface, color, (curr_x + 4, foot_rect.centery), 5)
                curr_x += 14
            elif kind == "halo":
                pygame.draw.circle(surface, color, (curr_x + 5, foot_rect.centery), 6, width=2)
                curr_x += 16

            txt_surf = self.font_small.render(label, True, self.TEXT_MUTED)
            surface.blit(txt_surf, (curr_x, foot_rect.centery - txt_surf.get_height() // 2))
            curr_x += txt_surf.get_width() + 18

    def _draw_translucent_line(self, surface, start_pos, end_pos, color, alpha, width):
        """Draw antialiased line with transparency."""
        min_x = min(start_pos[0], end_pos[0]) - width - 2
        max_x = max(start_pos[0], end_pos[0]) + width + 2
        min_y = min(start_pos[1], end_pos[1]) - width - 2
        max_y = max(start_pos[1], end_pos[1]) + width + 2

        w = max(1, max_x - min_x)
        h = max(1, max_y - min_y)

        line_surf = pygame.Surface((w, h), pygame.SRCALPHA)
        local_start = (start_pos[0] - min_x, start_pos[1] - min_y)
        local_end = (end_pos[0] - min_x, end_pos[1] - min_y)

        rgba = (color[0], color[1], color[2], min(255, max(0, alpha)))
        pygame.draw.line(line_surf, rgba, local_start, local_end, width)
        surface.blit(line_surf, (min_x, min_y))

    def _draw_glow(self, surface, center, radius, rgba):
        """Draw subtle translucent glow circle around a node."""
        surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(surf, rgba, (radius, radius), radius)
        surface.blit(surf, (center[0] - radius, center[1] - radius))

    def _draw_hover_tooltip(
        self,
        surface: pygame.Surface,
        network: NeuralNetwork,
        observation: np.ndarray,
        mouse_pos,
    ):
        """Display tooltip with neuron activation and bias details on hover."""
        mx, my = mouse_pos
        hovered_info = None

        # Check input nodes
        for i, (x, y) in enumerate(self.node_positions.get("input", [])):
            if math.hypot(mx - x, my - y) <= 12:
                meta = self.FEATURE_LABELS[i]
                val = float(observation.flatten()[i])
                hovered_info = [
                    f"Layer: Input Node {i}",
                    f"Feature: {meta['name']}",
                    f"Raw Value: {val:+.4f} {meta['unit']}",
                ]
                break

        # Check hidden nodes
        if not hovered_info:
            hidden_acts = network.A1[0] if hasattr(network, "A1") and network.A1 is not None else np.zeros(network.hidden_size)
            for h, (x, y) in enumerate(self.node_positions.get("hidden", [])):
                if math.hypot(mx - x, my - y) <= 10:
                    act_val = float(hidden_acts[h])
                    bias_val = float(network.b1[0, h])
                    hovered_info = [
                        f"Layer: Hidden Node {h}",
                        f"ReLU Activation: {act_val:.4f}",
                        f"Bias: {bias_val:+.4f}",
                        f"Status: {'Active (Firing)' if act_val > 0 else 'Inactive (0.0)'}",
                    ]
                    break

        # Check output nodes
        if not hovered_info:
            output_acts = network.A2[0] if hasattr(network, "A2") and network.A2 is not None else np.zeros(network.output_size)
            for o, (x, y) in enumerate(self.node_positions.get("output", [])):
                if math.hypot(mx - x, my - y) <= 14:
                    prob = float(output_acts[o])
                    bias_val = float(network.b2[0, o])
                    hovered_info = [
                        f"Layer: Output Node {o}",
                        f"Action: {self.ACTION_LABELS[o]}",
                        f"Probability: {prob * 100:.2f}%",
                        f"Bias: {bias_val:+.4f}",
                    ]
                    break

        if hovered_info:
            lines = [self.font_small.render(txt, True, (255, 255, 255)) for txt in hovered_info]
            max_w = max(l.get_width() for l in lines) + 20
            total_h = sum(l.get_height() for l in lines) + 14

            tip_x = min(surface.get_width() - max_w - 10, mx + 15)
            tip_y = min(surface.get_height() - total_h - 10, my + 15)

            tip_rect = pygame.Rect(tip_x, tip_y, max_w, total_h)
            pygame.draw.rect(surface, (18, 22, 32), tip_rect, border_radius=4)
            pygame.draw.rect(surface, self.TEXT_ACCENT, tip_rect, width=1, border_radius=4)

            curr_y = tip_y + 7
            for line in lines:
                surface.blit(line, (tip_x + 10, curr_y))
                curr_y += line.get_height()


def run_simulation_with_visualizer(
    network: NeuralNetwork,
    episodes: int = 5,
    max_steps: int = 500,
    fps: int = 60,
    weights_path: str = "best_cartpole_model.npz",
):
    """
    Run CartPole-v1 simulation alongside the real-time Neural Network Decision Display.
    The original Cart-Pole visual simulation remains 100% pixel-intact on the left,
    while the decision display renders in sync on the right.
    """
    pygame.init()
    pygame.display.set_caption("CartPole-v1 — Real-Time Neural Network Decision Display")

    # Layout geometry: 580 left (CartPole), 740 right (Decision Display)
    cartpole_w = 580
    cartpole_render_h = 360
    panel_w = 740
    window_w = cartpole_w + panel_w
    window_h = 600

    screen = pygame.display.set_mode((window_w, window_h))
    clock = pygame.time.Clock()

    # Create CartPole environment with rgb_array to render pixel-identical visuals
    env = gym.make("CartPole-v1", render_mode="rgb_array")

    # Neural network decision visualizer
    vis_rect = pygame.Rect(cartpole_w, 0, panel_w, window_h)
    visualizer = NetworkVisualizer(vis_rect)

    font_ui = get_font(13, bold=False)
    font_bold = get_font(14, bold=True)

    running = True
    paused = False

    print("=" * 60)
    print("  CartPole Decision Display Running")
    print("  Controls:")
    print("    [SPACE]   Pause / Resume simulation")
    print("    [R]       Reset episode")
    print("    [ESC]     Quit")
    print("=" * 60)

    for ep in range(1, episodes + 1):
        if not running:
            break

        observation, _ = env.reset()
        step = 0
        terminated = False
        truncated = False

        while running and not (terminated or truncated) and step < max_steps:
            mouse_pos = pygame.mouse.get_pos()

            # Handle Pygame window events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_r:
                        terminated = True

            if paused:
                # Keep displaying current frame while paused
                clock.tick(fps)
                continue

            # Model inference (uses existing forward/act pipeline without mutation)
            action = network.act(observation)

            # Step physical environment
            next_observation, reward, terminated, truncated, _ = env.step(action)
            step += 1

            # Fetch exact Gymnasium visual frame
            gym_frame = env.render()
            if gym_frame is not None:
                # Convert (H, W, C) numpy array to Pygame Surface
                surf_gym = pygame.surfarray.make_surface(np.transpose(gym_frame, (1, 0, 2)))
                # Scale cleanly to fit left panel maintaining clean aspect ratio
                surf_scaled = pygame.transform.smoothscale(surf_gym, (cartpole_w - 30, cartpole_render_h))
            else:
                surf_scaled = None

            # Render overall scene
            screen.fill((12, 15, 22))

            # ----------------- Left Panel: CartPole Visual Simulation -----------------
            # Environment Card
            env_card = pygame.Rect(12, 12, cartpole_w - 24, window_h - 24)
            pygame.draw.rect(screen, (20, 24, 34), env_card, border_radius=6)
            pygame.draw.rect(screen, (40, 50, 70), env_card, width=1, border_radius=6)

            # Title
            title_txt = font_bold.render("SIMULATION: Gymnasium CartPole-v1", True, (240, 243, 248))
            screen.blit(title_txt, (env_card.left + 15, env_card.top + 12))

            status_color = (46, 213, 115) if not paused else (255, 159, 26)
            status_txt = font_ui.render("Status: LIVE" if not paused else "Status: PAUSED", True, status_color)
            screen.blit(status_txt, (env_card.right - status_txt.get_width() - 15, env_card.top + 14))

            # Blit original cartpole simulation surface
            if surf_scaled is not None:
                screen.blit(surf_scaled, (env_card.left + 5, env_card.top + 40))

            # Simulation Telemetry box below cartpole
            telemetry_y = env_card.top + cartpole_render_h + 50
            tele_rect = pygame.Rect(env_card.left + 8, telemetry_y, env_card.width - 16, env_card.bottom - telemetry_y - 10)
            pygame.draw.rect(screen, (16, 20, 28), tele_rect, border_radius=4)
            pygame.draw.rect(screen, (34, 43, 60), tele_rect, width=1, border_radius=4)

            # Draw observation readings in 2 columns
            labels_col1 = [
                ("Cart Pos (x):", f"{observation[0]:+.3f} m"),
                ("Cart Vel (v):", f"{observation[1]:+.3f} m/s"),
            ]
            labels_col2 = [
                ("Pole Angle (θ):", f"{math.degrees(observation[2]):+.2f}° ({observation[2]:+.3f} rad)"),
                ("Pole Vel (ω):", f"{math.degrees(observation[3]):+.2f}°/s"),
            ]

            t_y = tele_rect.top + 12
            for lbl, val in labels_col1:
                lbl_s = font_ui.render(lbl, True, (130, 145, 165))
                val_s = font_bold.render(val, True, (220, 230, 242))
                screen.blit(lbl_s, (tele_rect.left + 12, t_y))
                screen.blit(val_s, (tele_rect.left + 115, t_y))
                t_y += 24

            t_y = tele_rect.top + 12
            for lbl, val in labels_col2:
                lbl_s = font_ui.render(lbl, True, (130, 145, 165))
                val_s = font_bold.render(val, True, (220, 230, 242))
                screen.blit(lbl_s, (tele_rect.left + 260, t_y))
                screen.blit(val_s, (tele_rect.left + 370, t_y))
                t_y += 24

            # Control hints at bottom of telemetry box
            hint_txt = font_ui.render("[SPACE] Pause/Resume   |   [R] Reset Episode   |   [ESC] Exit", True, (100, 115, 135))
            screen.blit(hint_txt, (tele_rect.left + 12, tele_rect.bottom - 24))

            # Divider line between left & right panels
            pygame.draw.line(screen, (44, 55, 77), (cartpole_w, 0), (cartpole_w, window_h), width=2)

            # ----------------- Right Panel: Neural Decision Display -----------------
            visualizer.draw(
                surface=screen,
                network=network,
                observation=observation,
                action=action,
                step=step,
                episode=ep,
                mouse_pos=mouse_pos,
            )

            # Update display and maintain 60 FPS
            pygame.display.flip()
            clock.tick(fps)

            observation = next_observation

        if running:
            print(f"Episode {ep} finished after {step} steps.")

    env.close()
    pygame.quit()


if __name__ == "__main__":
    import os

    # Load weights or fallback to fresh network
    weights_file = "best_cartpole_model.npz"
    net = NeuralNetwork(input_size=4, hidden_size=16, output_size=2)
    if os.path.exists(weights_file):
        print(f"Loading trained model weights from '{weights_file}'...")
        net.load(weights_file)
    else:
        print(f"Warning: '{weights_file}' not found. Initializing random weights.")

    run_simulation_with_visualizer(net, episodes=5, max_steps=500)
