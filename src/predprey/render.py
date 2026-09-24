"""Pygame viewer for the predator-prey environment."""

import numpy as np
import pygame

from predprey.physics import ARENA_SIZE, AgentState

WINDOW_SIZE = 600                    # pixels (square window)
SCALE = WINDOW_SIZE / ARENA_SIZE     # pixels per world unit

BACKGROUND = (245, 243, 238)
BORDER = (60, 60, 60)
TEXT = (30, 30, 30)
AGENT_COLOURS = {
    "predator": (214, 69, 65),   # red
    "prey": (52, 110, 200),      # blue
}
DEFAULT_COLOUR = (120, 120, 120)

VELOCITY_LINE_SCALE = 0.3  # how long the velocity line is, in seconds of travel


def world_to_screen(position: np.ndarray) -> tuple[int, int]:
    """Convert world coordinates (+y up) to screen pixels (+y down)."""
    x = position[0] * SCALE
    y = WINDOW_SIZE - position[1] * SCALE
    return round(x), round(y)


class Renderer:
    def __init__(self, fps: int = 10):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        pygame.display.set_caption("Predator vs Prey")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 26)
        self.fps = fps
        self.is_open = True

    def draw(self, states: dict[str, AgentState], step: int, caught: bool) -> None:
        # Handle window events every frame, or the OS thinks we've frozen
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        self.screen.fill(BACKGROUND)
        pygame.draw.rect(self.screen, BORDER, self.screen.get_rect(), width=3)

        for name, state in states.items():
            colour = AGENT_COLOURS.get(name, DEFAULT_COLOUR)
            centre = world_to_screen(state.position)
            radius = max(1, round(state.radius * SCALE))
            pygame.draw.circle(self.screen, colour, centre, radius)

            # Line showing where the agent is heading
            tip = state.position + state.velocity * VELOCITY_LINE_SCALE
            pygame.draw.line(self.screen, BORDER, centre, world_to_screen(tip), width=2)

        label = f"step {step}" + ("   CAUGHT!" if caught else "")
        self.screen.blit(self.font.render(label, True, TEXT), (10, 10))

        pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self) -> None:
        if self.is_open:
            pygame.quit()
            self.is_open = False