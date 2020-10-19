from pathlib import Path

import gym

from agent import agent_spec

# Path to the scenario to test
scenario_path = (
    # Path(__file__).parent / "../../dataset_public/simple_loop/simpleloop_a"
    # Path(__file__).parent / "../../dataset_public/sharp_loop/sharploop_a"
    # Path(__file__).parent / "../../dataset_public/all_loop/all_loop_a"
    # Path(__file__).parent / "../../dataset_public/merge_loop/merge_a"
    # Path(__file__).parent / "../../dataset_public/intersection_loop/its_a"
    # Path(__file__).parent / "../../dataset_public/roundabout_loop/roundabout_a"
    # Path(__file__).parent / "../../dataset_public/demo/demo1_4_triangle"
    Path(__file__).parent / "../../dataset_public/mixed_loop/its_merge_a"
    # Path(__file__).parent / "../../dataset_public/mixed_loop/roundabout_its_a"
    # Path(__file__).parent / "../../dataset_public/mixed_loop/roundabout_merge_a"
    # Path(__file__).parent / "../../dataset_public/simple_loop/demo_simpleloop_1"
).resolve()

AGENT_ID = "Agent-007"


def main():

    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=[scenario_path],
        agent_specs={AGENT_ID: agent_spec},
        # set headless to false if u want to use envision
        headless=False,
        visdom=False,
        seed=30,
    )

    agent = agent_spec.build_agent()

    while True:
        step = 0
        observations = env.reset()
        total_reward = 0.0
        dones = {"__all__": False}

        while not dones["__all__"]:
            step += 1
            agent_obs = observations[AGENT_ID]

            #print(agent_obs['speed'])
            agent_action = agent.act(agent_obs)
            observations, rewards, dones, _ = env.step({AGENT_ID: agent_action})

            total_reward += rewards[AGENT_ID]
            if step == 1500:
                break

        print("Accumulated reward:", total_reward)
        print(step)
        print(_['Agent-007']['env_obs'].events.reached_goal)
        print()
    env.close()


if __name__ == "__main__":
    main()
