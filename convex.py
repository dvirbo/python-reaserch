"""
Convex Hull Algorithms - sending both the algorithm and the output type as argument.

Author: Dvir Borochov
Since: 2024-05
"""

from typing import Callable, Union, List, Tuple, Dict
import math


Point = Tuple[float, float]
Polygon = List[Point]
PointsDictionary = Dict[str, Point]
InputData = Union[Polygon, PointsDictionary]
OutputData = Union[Polygon, PointsDictionary]


def convex_hull(algorithm: Callable, input_data: InputData, output_type: str = "polygon") -> OutputData:
    """
    Finds the convex hull of the given input data using the specified algorithm and output type.

    Args:
        algorithm (Callable): The algorithm to use for finding the convex hull.
        input_data (InputData): The input data, either a list of points or a dictionary of points.
        output_type (str, optional): The desired output type, either "polygon" or "dictionary". Defaults to "polygon".

    Returns:
        OutputData: The convex hull in the specified output type.

    Examples:
        >>> convex_hull(algorithm=graham_scan, input_data=[(0, 3), (2, 2), (1, 1), (2, 1), (3, 0), (0, 0), (3, 3)], output_type="polygon")
        [(3, 0), (0, 0), (0, 3), (3, 3)]
        >>> convex_hull(algorithm=graham_scan, input_data={"a": (0, 3), "b": (2, 2), "c": (1, 1), "d": (2, 1), "e": (3, 0), "f": (0, 0), "g": (3, 3)}, output_type="dictionary")
        {"e": (3, 0), "f": (0, 0), "a": (0, 3), "g": (3, 3)}
        >>> convex_hull(algorithm=jarvis_march, input_data=[(0, 3), (2, 2), (1, 1), (2, 1), (3, 0), (0, 0), (3, 3)], output_type="polygon")
        [(3, 0), (0, 0), (0, 3), (3, 3)]
        >>> convex_hull(algorithm=jarvis_march, input_data={"a": (0, 3), "b": (2, 2), "c": (1, 1), "d": (2, 1), "e": (3, 0), "f": (0, 0), "g": (3, 3)}, output_type="dictionary")
        {"e": (3, 0), "f": (0, 0), "a": (0, 3), "g": (3, 3)}
    """
    if isinstance(input_data, list):
        points = input_data
    else:
        points = list(input_data.values())

    convex_hull_points = algorithm(points)

    if output_type == "polygon":
        return convex_hull_points
    else:
        return {chr(97 + i): point for i, point in enumerate(convex_hull_points)}


def polar_angle(p0: Point, p1: Point) -> float:
    """
    Calculates the polar angle between two points.

    Parameters:
    p0 (Point): The first point.
    p1 (Point): The second point.

    Returns:
    float: The polar angle between the two points in radians.
    """
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    return math.atan2(dy, dx)


def distance(p0: Point, p1: Point) -> float:
    """
    Calculate the Euclidean distance between two points.

    Args:
        p0 (Point): The first point.
        p1 (Point): The second point.

    Returns:
        float: The Euclidean distance between the two points.
    """
    return (p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2


def graham_scan(points: Polygon) -> Polygon:
    """
    Implements the Graham Scan algorithm for finding the convex hull of a set of points.

    Args:
        points (Polygon): A list of points.

    Returns:
        Polygon: The convex hull as a list of points.
    """

    # Step 1: Find the point with the lowest y-coordinate (and leftmost if tie)
    start = min(points, key=lambda p: (p[1], p[0]))

    # Step 2: Sort the points by polar angle with respect to the start point
    sorted_points = sorted(
        points, key=lambda p: (polar_angle(start, p), -distance(start, p))
    )

    # Step 3: Create the convex hull using a stack
    hull = [sorted_points[0], sorted_points[1]]
    for point in sorted_points[2:]:
        while (
            len(hull) > 1
            and (hull[-1][0] - hull[-2][0]) * (point[1] - hull[-1][1])
            - (hull[-1][1] - hull[-2][1]) * (point[0] - hull[-1][0])
            <= 0
        ):
            hull.pop()
        hull.append(point)

    return hull


def jarvis_march(points: Polygon) -> Polygon:
    """
    Implements the Jarvis March algorithm for finding the convex hull of a set of points.

    Args:
        points (Polygon): A list of points.

    Returns:
        Polygon: The convex hull as a list of points.
    """

    # Step 1: Find the leftmost point
    leftmost_point = min(points, key=lambda p: (p[0], p[1]))

    # Step 2: Initialize the convex hull with the leftmost point
    convex_hull = [leftmost_point]
    current_point = leftmost_point

    while True:
        # Step 3: Find the next point in the convex hull
        next_point = None
        max_angle = float("-inf")
        max_distance = float("-inf")
        for point in points:
            if point != current_point:
                angle = polar_angle(current_point, point)
                dist = distance(current_point, point)
                if angle > max_angle or (angle == max_angle and dist > max_distance):
                    max_angle = angle
                    max_distance = dist
                    next_point = point

        # Step 4: Add the next point to the convex hull
        convex_hull.append(next_point)
        current_point = next_point

        # Step 5: Check if we've completed the convex hull
        if next_point == convex_hull[0]:
            break

    return convex_hull


if __name__ == "__main__":
    # Example usage
    points = [(2,0),(6,0), (6,4), (8,4), (8,6),(0,6),(0,4), (2,4)]
    dict_points = {
        "a": (2,0),
        "b": (6, 0),
        "c": (6, 4),
        "d": (8, 4),
        "e": (8, 6),
        "f": (0, 6),
        "g": (2, 4)
    }

    # 1. Graham Scan algorithm with list of points as input and polygon as output
    convex_hull_points = convex_hull(algorithm=graham_scan, input_data=points, output_type="polygon")
    print("Graham Scan - List of points as input, polygon as output:")
    print(convex_hull_points)

    # 2. Graham Scan algorithm with list of points as input and dictionary as output
    convex_hull_dict = convex_hull(algorithm=graham_scan, input_data=points, output_type="dictionary")
    print("Graham Scan - List of points as input, dictionary as output:")
    print(convex_hull_dict)

    # 3. Graham Scan algorithm with dictionary of points as input and polygon as output
    convex_hull_points = convex_hull(algorithm=graham_scan, input_data=dict_points, output_type="polygon")
    print("Graham Scan - Dictionary of points as input, polygon as output:")
    print(convex_hull_points)

    # 4. Graham Scan algorithm with dictionary of points as input and dictionary as output
    convex_hull_dict = convex_hull(algorithm=graham_scan, input_data=dict_points, output_type="dictionary")
    print("Graham Scan - Dictionary of points as input, dictionary as output:")
    print(convex_hull_dict)

    # 5. Jarvis March algorithm with list of points as input and polygon as output
    convex_hull_points = convex_hull(algorithm=jarvis_march, input_data=points, output_type="polygon")
    print("Jarvis March - List of points as input, polygon as output:")
    print(convex_hull_points)

    # 6. Jarvis March algorithm with list of points as input and dictionary as output
    convex_hull_dict = convex_hull(algorithm=jarvis_march, input_data=points, output_type="dictionary")
    print("Jarvis March - List of points as input, dictionary as output:")
    print(convex_hull_dict)

    # 7. Jarvis March algorithm with dictionary of points as input and polygon as output
    convex_hull_points = convex_hull(algorithm=jarvis_march, input_data=dict_points, output_type="polygon")
    print("Jarvis March - Dictionary of points as input, polygon as output:")
    print(convex_hull_points)

    # 8. Jarvis March algorithm with dictionary of points as input and dictionary as output
    convex_hull_dict = convex_hull(algorithm=jarvis_march, input_data=dict_points, output_type="dictionary")
    print("Jarvis March - Dictionary of points as input, dictionary as output:")
    print(convex_hull_dict)
