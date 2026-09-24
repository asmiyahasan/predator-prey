"""Pygame viewer for the predator-prey environment."""

import numpy as np
import pygame

from predprey.physics import ARENA_SIZE, NO_OBSTACLES, AgentState
from predprey.sensors import RAY_DIRECTIONS, cast_rays

WINDOW_SIZE = 600                    # pixels (square window)
SCALE = WINDOW_SIZE / ARENA_SIZE     # pixels per world unit

BACKGROUND = (245, 243, 238)
BORDER = (60, 60, 60)
OBSTACLE = (90, 90, 95)
TEXT = (30, 30, 30)
SIGHT_LINE = (150, 150, 150)
AGENT_COLOURS = {
    "predator": (214, 69, 65),   # red
    "prey": (52, 110, 200),      # blue
}
RAY_COLOURS = {
    "predator": (235, 170, 165),
    "prey": (165, 190, 230),
}
DEFAULT_COLOUR = (120, 120, 120)

VELOCITY_LINE_SCALE = 0.3  # how long the velocity line is, in seconds of travel


def world_to_screen(position: np.ndarray) -> tuple[int, int]:
    """Convert world coordinates (+y up) to screen pixels (+y down)."""
    x = position[0] * SCALE
    y = WINDOW_SIZE - position[1] * SCALE
    return round(x), round(y)


def rect_to_screen(bounds: np.ndarray) -> pygame.Rect:
    """Convert [x_min, y_min, x_max, y_max] world bounds to a pygame Rect."""
    left, top = world_to_screen(np.array([bounds[0], bounds[3]]))   # top-left corner
    right, bottom = world_to_screen(np.array([bounds[2], bounds[1]]))
    return pygame.Rect(left, top, right - left, bottom - top)


class Renderer:
    def __init__(self, fps: int = 10, show_rays: bool = False):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        pygame.display.set_caption("Predator vs Prey")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 26)
        self.fps = fps
        self.show_rays = show_rays
        self.is_open = True

    def draw(
        self,
        states: dict[str, AgentState],
        step: int,
        caught: bool,
        obstacles: np.ndarray = NO_OBSTACLES,
        visible: bool = True,
    ) -> None:
        # Handle window events every frame, or the OS thinks we've frozen
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                self.show_rays = not self.show_rays  # press R to toggle rays

        self.screen.fill(BACKGROUND)

        for bounds in obstacles:
            pygame.draw.rect(self.screen, OBSTACLE, rect_to_screen(bounds))

        if self.show_rays:
            for name, state in states.items():
                colour = RAY_COLOURS.get(name, DEFAULT_COLOUR)
                distances = cast_rays(state.position, obstacles)
                for direction, distance in zip(RAY_DIRECTIONS, distances):
                    end = state.position + direction * distance
                    pygame.draw.line(self.screen, colour, world_to_screen(state.position),
                                     world_to_screen(end), width=1)

        # A thin line between the agents whenever they can see each other
        if visible and len(states) == 2:
            a, b = (s.position for s in states.values())
            pygame.draw.line(self.screen, SIGHT_LINE, world_to_screen(a), world_to_screen(b), 1)

        for name, state in states.items():
            colour = AGENT_COLOURS.get(name, DEFAULT_COLOUR)
            centre = world_to_screen(state.position)
            radius = max(1, round(state.radius * SCALE))
            pygame.draw.circle(self.screen, colour, centre, radius)

            # Line showing where the agent is heading
            tip = state.position + state.velocity * VELOCITY_LINE_SCALE
            pygame.draw.line(self.screen, BORDER, centre, world_to_screen(tip), width=2)

        pygame.draw.rect(self.screen, BORDER, self.screen.get_rect(), width=3)

        label = f"step {step}" + ("" if visible else "   hidden") + ("   CAUGHT!" if caught else "")
        self.screen.blit(self.font.render(label, True, TEXT), (10, 10))

        pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self) -> None:
        if self.is_open:
            pygame.quit()
            self.is_open = False
