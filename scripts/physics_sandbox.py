import numpy as np

from predprey.physics import ARENA_SIZE, NUM_ACTIONS, AgentState, step_agent, is_caught

RIGHT = 4
STAY = 0


def print_state(step: int, action: int, agent: AgentState) -> None:
    speed = np.linalg.norm(agent.velocity)
    print(
        f"step {step:3d} | action {action} | "
        f"pos ({agent.position[0]:5.2f}, {agent.position[1]:5.2f}) | "
        f"vel ({agent.velocity[0]:5.2f}, {agent.velocity[1]:5.2f}) | "
        f"speed {speed:4.2f}"
    )


# --- Test 1: hold "right" -> speed up, cap at max_speed, stop at the wall ---
print("=== Test 1: always move right ===")
agent = AgentState(position=[5.0, 5.0])
for step in range(40):
    step_agent(agent, RIGHT)
    print_state(step, RIGHT, agent)

print(f"\nExpected final x: {ARENA_SIZE - agent.radius}, x-velocity: 0.0")

# --- Test 2: start moving, then stay still -> drag slows it down ---
print("\n=== Test 2: drag slows the agent ===")
agent = AgentState(position=[5.0, 5.0], velocity=[2.0, 0.0])
for step in range(20):
    step_agent(agent, STAY)
    print_state(step, STAY, agent)

# --- Test 3: random walk -> never leaves arena, never too fast ---
print("\n=== Test 3: 1,000 random steps ===")
rng = np.random.default_rng(seed=42)
agent = AgentState(position=[5.0, 5.0])
low, high = agent.radius, ARENA_SIZE - agent.radius

max_seen_speed = 0.0
stayed_inside = True
for _ in range(1000):
    step_agent(agent, rng.integers(NUM_ACTIONS))
    max_seen_speed = max(max_seen_speed, np.linalg.norm(agent.velocity))
    if np.any(agent.position < low) or np.any(agent.position > high):
        stayed_inside = False

print(f"Stayed inside arena: {stayed_inside}")
print(f"Highest speed seen:  {max_seen_speed:.4f} (max_speed = {agent.max_speed})")



# --- Test 4: catch detection ---
print("\n=== Test 4: catch detection ===")
predator = AgentState(position=[2.0, 2.0])
prey = AgentState(position=[8.0, 8.0])
print(f"Far apart:        {is_caught(predator, prey)}  (expected False)")

prey.position[:] = [2.3, 2.0]   # 0.3 apart, radii sum to 0.6
print(f"Overlapping:      {is_caught(predator, prey)}  (expected True)")

prey.position[:] = [2.6, 2.0]   # exactly 0.6 apart
print(f"Exactly touching: {is_caught(predator, prey)}  (expected True)")

prey.position[:] = [2.61, 2.0]  # just beyond 0.6
print(f"Just apart:       {is_caught(predator, prey)}  (expected False)")