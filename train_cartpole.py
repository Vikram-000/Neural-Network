import argparse
import time
import gymnasium as gym
import numpy as np
from basic_neural_network import NeuralNetwork

def evaluate_network(network: NeuralNetwork, env: gym.Env, episodes: int = 2, max_steps: int = 500) -> float:
    total_steps = 0
    for _ in range(episodes):
        observation, _ = env.reset()
        episode_steps = 0
        terminated = False
        truncated = False

        while not (terminated or truncated) and episode_steps < max_steps:
            action = network.act(observation)
            observation, reward, terminated, truncated, _ = env.step(action)
            episode_steps += 1

        total_steps += episode_steps

    return total_steps / episodes

def verify_champion(network: NeuralNetwork, num_trials: int = 5, max_steps: int = 500) -> float:
    env = gym.make("CartPole-v1")
    scores = []
    for seed in range(num_trials):
        observation, _ = env.reset(seed=seed * 101)
        steps = 0
        terminated = False
        truncated = False
        while not (terminated or truncated) and steps < max_steps:
            action = network.act(observation)
            observation, _, terminated, truncated, _ = env.step(action)
            steps += 1
        scores.append(steps)
    env.close()
    return float(np.mean(scores))

def render_network(network: NeuralNetwork, episodes: int = 3, max_steps: int = 500):
    print(f"\n[Visual Demo] Rendering trained network for {episodes} episodes...")
    env = gym.make("CartPole-v1", render_mode="human")
    for ep in range(1, episodes + 1):
        observation, _ = env.reset()
        steps = 0
        terminated = False
        truncated = False

        while not (terminated or truncated) and steps < max_steps:
            action = network.act(observation)
            observation, _, terminated, truncated, _ = env.step(action)
            steps += 1

        print(f"  Episode {ep}: Balanced for {steps} steps.")
    env.close()
    print("[Visual Demo] Finished.\n")

def train_genetic_algorithm(
    population_size: int = 50,
    hidden_size: int = 16,
    max_generations: int = 50,
    target_fitness: float = 500.0,
    elite_ratio: float = 0.1,
    mutation_rate: float = 0.2,
    mutation_strength: float = 0.3,
    eval_episodes: int = 2,
    save_path: str = "best_cartpole_model.npz",
    render_after_training: bool = False,
):
    print("=" * 60)
    print("  CartPole Neural Network Training (Genetic Algorithm)")
    print("=" * 60)
    print(f"Population Size   : {population_size}")
    print(f"Network Topology  : 4 Inputs -> {hidden_size} Hidden -> 2 Outputs")
    print(f"Target Fitness    : {target_fitness} steps")
    print(f"Elite Ratio       : {elite_ratio * 100:.0f}%")
    print(f"Mutation Rate     : {mutation_rate} (scale: {mutation_strength})")
    print("=" * 60)

    train_env = gym.make("CartPole-v1")

    population = [
        NeuralNetwork(input_size=4, hidden_size=hidden_size, output_size=2)
        for _ in range(population_size)
    ]

    elite_count = max(1, int(population_size * elite_ratio))
    start_time = time.time()
    best_overall_network = None
    best_overall_fitness = 0.0

    for generation in range(1, max_generations + 1):
        scored_population = []
        for network in population:
            fitness = evaluate_network(network, train_env, episodes=eval_episodes)
            scored_population.append((fitness, network))

        scored_population.sort(key=lambda item: item[0], reverse=True)

        gen_best_fitness, gen_best_network = scored_population[0]
        gen_avg_fitness = sum(item[0] for item in scored_population) / population_size

        if gen_best_fitness > best_overall_fitness:
            best_overall_fitness = gen_best_fitness
            best_overall_network = gen_best_network.clone()

        print(
            f"Gen {generation:02d}/{max_generations:02d} | "
            f"Best: {gen_best_fitness:5.1f} | "
            f"Avg: {gen_avg_fitness:5.1f} | "
            f"Overall Best: {best_overall_fitness:5.1f}"
        )

        if gen_best_fitness >= target_fitness:
            avg_verification = verify_champion(gen_best_network, num_trials=5)
            print(f"\n[Target Reached] Generation {generation} achieved {gen_best_fitness:.1f} steps!")
            print(f"[Verification] 5-seed verification average: {avg_verification:.1f} steps.")
            if avg_verification >= 200.0:  # Requirement: >= 200 frames
                print(">>> CartPole successfully solved! <<<")
                best_overall_network = gen_best_network.clone()
                break

        new_population = [scored_population[i][1].clone() for i in range(elite_count)]

        parent_pool_size = max(elite_count, int(population_size * 0.4))
        parent_pool = [scored_population[i][1] for i in range(parent_pool_size)]

        while len(new_population) < population_size:
            parent = parent_pool[np.random.randint(0, len(parent_pool))]
            child = parent.clone()
            child.mutate(mutation_rate=mutation_rate, mutation_strength=mutation_strength)
            new_population.append(child)

        population = new_population

    train_env.close()
    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.2f} seconds.")

    if best_overall_network is not None:
        best_overall_network.save(save_path)
        print(f"Saved champion network weights to '{save_path}'.")

        final_test_score = verify_champion(best_overall_network, num_trials=10)
        print(f"Final 10-episode generalization benchmark: {final_test_score:.1f} / 500.0 steps")

        if render_after_training:
            render_network(best_overall_network)

    return best_overall_network

def main():
    parser = argparse.ArgumentParser(description="Train Neural Network on CartPole-v1 using Genetic Algorithm")
    parser.add_argument("--pop-size", type=int, default=50, help="Population size (default: 50)")
    parser.add_argument("--hidden-size", type=int, default=16, help="Hidden layer size (default: 16)")
    parser.add_argument("--generations", type=int, default=30, help="Maximum generations (default: 30)")
    parser.add_argument("--target", type=float, default=500.0, help="Target fitness threshold (default: 500)")
    parser.add_argument("--render", action="store_true", help="Render champion network after training")
    parser.add_argument("--test-only", action="store_true", help="Load saved model and render without training")
    parser.add_argument("--weights", type=str, default="best_cartpole_model.npz", help="Weights file path")
    args = parser.parse_args()

    if args.test_only:
        network = NeuralNetwork(input_size=4, hidden_size=args.hidden_size, output_size=2)
        try:
            network.load(args.weights)
            print(f"Loaded weights from {args.weights}")
            render_network(network, episodes=5)
        except FileNotFoundError:
            print(f"Error: Model file '{args.weights}' not found. Please train first!")
    else:
        train_genetic_algorithm(
            population_size=args.pop_size,
            hidden_size=args.hidden_size,
            max_generations=args.generations,
            target_fitness=args.target,
            save_path=args.weights,
            render_after_training=args.render,
        )

if __name__ == "__main__":
    main()
