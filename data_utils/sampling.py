import math
import random
import numpy as np


def create_noniid_users(train_set, args, alpha=100):
    """
    Sample non-I.I.D client data from CIFAR10 dataset.

    Args:
        train_set: Training dataset containing data and targets.
        args: Arguments containing dataset parameters (e.g., number of clients, classes).
        alpha: Dirichlet distribution parameter controlling data heterogeneity.

    Returns:
        train_dict: Dictionary where keys are client IDs and values are lists of data indices.
    """

    # Original resource: https://github.com/epfml/federated-learning-public-code/blob/master/codes/FedDF-code/pcode/datasets/partition_data.py
    def build_non_iid_by_dirichlet(
            random_state, indices2targets, alpha, num_classes, num_indices, num_clients
    ):
        # Auxiliary workers to assist partitioning
        aux_workers = 10
        assert aux_workers <= num_clients

        # Shuffle target indices
        random_state.shuffle(indices2targets)

        # Partition indices into groups for clients
        num_groups = math.ceil(num_clients / aux_workers)
        group_sizes = [aux_workers if i < num_groups - 1 else num_clients - aux_workers * (num_groups - 1) for i in range(num_groups)]

        # Split targets into groups based on group sizes
        split_ratios = [size / num_clients for size in group_sizes]
        from_index = 0
        grouped_targets = []
        for ratio in split_ratios:
            to_index = from_index + int(ratio * num_indices)
            grouped_targets.append(indices2targets[from_index:to_index])
            from_index = to_index

        # Allocate data indices to clients
        client_batches = []
        for targets in grouped_targets:
            targets = np.array(targets)
            num_targets = len(targets)

            # Use auxiliary workers for this subset
            workers = min(aux_workers, num_clients)
            num_clients -= aux_workers

            samples_per_client = num_targets // workers

            # Ensure minimum class distribution size
            min_size = 0
            while min_size < int(0.5 * num_targets / workers):
                client_batches_group = [[] for _ in range(workers)]
                
                for cls in range(num_classes):
                    class_indices = np.where(targets[:, 1] == cls)[0]
                    class_indices = targets[class_indices, 0]

                    try:
                        proportions = random_state.dirichlet(np.repeat(alpha, workers))
                        proportions = np.array([
                            p * (len(batch) < num_targets / workers)
                            for p, batch in zip(proportions, client_batches_group)
                        ])
                        proportions /= proportions.sum()

                        split_indices = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
                        client_batches_group = [
                            batch + indices.tolist()
                            for batch, indices in zip(client_batches_group, np.split(class_indices, split_indices))
                        ]

                    except ZeroDivisionError:
                        pass

                # Balance batch sizes
                extra_data = []
                for batch in client_batches_group:
                    random.shuffle(batch)

                for i in range(workers):
                    if len(client_batches_group[i]) > samples_per_client:
                        excess = len(client_batches_group[i]) - samples_per_client
                        extra_data.extend(client_batches_group[i][-excess:])
                        client_batches_group[i] = client_batches_group[i][:-excess]

                for i in range(workers):
                    if len(client_batches_group[i]) < samples_per_client:
                        shortage = samples_per_client - len(client_batches_group[i])
                        client_batches_group[i].extend(extra_data[:shortage])
                        extra_data = extra_data[shortage:]

                min_size = min(len(batch) for batch in client_batches_group)

            client_batches += client_batches_group

        return {i: indices for i, indices in enumerate(client_batches)}

    # Determine the correct attribute for labels
    if hasattr(train_set, "labels"):  # For SVHN or similar datasets
        labels = train_set.labels
    elif hasattr(train_set, "targets"):  # For datasets like CIFAR
        labels = train_set.targets
    else:
        raise AttributeError("Dataset does not have 'labels' or 'targets' attribute.")
    
    # Generate non-IID client data
    dict_users = build_non_iid_by_dirichlet(
        random_state=np.random.RandomState(1),
        indices2targets=np.array([(idx, target) for idx, target in enumerate(labels)]),
        alpha=alpha,
        num_classes=args.num_classes,
        num_indices=len(train_set),
        num_clients=args.num_clients
    )

    # Filter train data indices
    train_dict = {
        client_id: [idx for idx in indices if idx < len(train_set)]
        for client_id, indices in dict_users.items()
    }

    # Print class distribution per client
    def count_classes(dataset, indices):
        class_counts = {}
        for idx in indices:
            label = labels[idx]
            class_counts[label] = class_counts.get(label, 0) + 1
        return class_counts

    for client_id, indices in train_dict.items():
        train_size = len(indices)
        train_class_counts = count_classes(train_set, indices)
        sorted_class_counts = dict(sorted(train_class_counts.items()))
        print(f"Client {client_id} training set size: {train_size}")
        print(f"Client {client_id} class distribution: {sorted_class_counts}")

    return train_dict
