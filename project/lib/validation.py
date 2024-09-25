import numpy as np

import torch

def validation_run(env, net, Actions, episodes=100, device="cpu", epsilon=0.02, commission=0.1):
    stats = {
        'episode_reward': [],
        'episode_steps': [],
        'order_profits': [],
        'order_steps': [],
    }
    if len(Actions) == 3:
        for episode in range(episodes):
            obs, _ = env.reset()

            total_reward = 0.0
            position = None
            position_steps = None
            episode_steps = 0

            while True:
                obs_v = torch.tensor(np.array([obs])).to(device)
                out_v = net(obs_v)

                action_idx = out_v.max(dim=1)[1].item()
                if np.random.random() < epsilon:
                    action_idx = env.action_space.sample()
                action = Actions(action_idx)

                close_price = env._state._cur_close()

                if action == Actions.Buy and position is None:
                    position = close_price
                    position_steps = 0
                elif action == Actions.Close and position is not None:
                    profit = close_price - position - (close_price + position) * commission / 100
                    profit = 100.0 * profit / position
                    stats['order_profits'].append(profit)
                    stats['order_steps'].append(position_steps)
                    position = None
                    position_steps = None

                obs, reward, done, _, _ = env.step(action_idx)
                total_reward += reward
                episode_steps += 1
                if position_steps is not None:
                    position_steps += 1
                if done:
                    if position is not None:
                        profit = close_price - position - (close_price + position) * commission / 100
                        profit = 100.0 * profit / position
                        stats['order_profits'].append(profit)
                        stats['order_steps'].append(position_steps)
                    break

            stats['episode_reward'].append(total_reward)
            stats['episode_steps'].append(episode_steps)

        return {key: np.mean(vals) for key, vals in stats.items()}
    else:
        for episode in range(episodes):
            obs, _ = env.reset()

            total_reward = 0.0
            position_steps = 0
            episode_steps = 0

            while True:
                obs_v = torch.tensor(np.array([obs])).to(device)
                out_v = net(obs_v)

                action_idx = out_v.max(dim=1)[1].item()
                if np.random.random() < epsilon:
                    action_idx = env.action_space.sample()
                action = Actions(action_idx)

                close_price = env._state._cur_close()
                if env._state.stocks_owned > 0:
                    position_steps += 1
                prev_capital = env._state.capital
                obs, reward, done, _, _ = env.step(action_idx)
                total_reward += reward
                episode_steps += 1
                if env._state.stocks_owned > 0:
                    position_steps += 1
                else:
                    position_steps = 0
                profit = env._state.capital - prev_capital
                stats['order_steps'].append(position_steps)
                if done:
                    profit += env._state._cur_close()*env._state.stocks_owned * (1 - env._state.commission_perc)
                    stats['order_profits'].append(profit)
                    break
                stats['order_profits'].append(profit)


            stats['episode_reward'].append(total_reward)
            stats['episode_steps'].append(episode_steps)

        return {key: np.mean(vals) for key, vals in stats.items()}
