from collections import defaultdict

# returns the ranking in ascending order
def build_sorted_shots(comparisons):
    graph = defaultdict(list)
    for item1, item2, result in comparisons:
        if result == 1:
            graph[item1].append(item2)
        else:
            graph[item2].append(item1)

    visited = set()
    ranking = []

    def dfs(node):
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor)
        ranking.append(node)

    for node in list(graph.keys()):
        if node not in visited:
            dfs(node)

    # ranking.reverse()
    return ranking


if __name__ == "__main__":
    # Example usage
    comparisons = [
        ('A', 'B', 1),
        ('B', 'C', 1),
        ('C', 'D', 1),
        ('D', 'E', 1),
        ('E', 'F', 1),
        ('F', 'G', 1),
        ('G', 'H', 1),
        ('H', 'I', 1),
        ('I', 'J', 1)
    ]
    ranking = build_sorted_shots(comparisons)
    print(ranking)

