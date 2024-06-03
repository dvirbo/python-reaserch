import logging
from shapely.geometry import (
    Polygon,
    LineString,
    Point,
    GeometryCollection,
    MultiLineString,
    MultiPolygon,
)
from shapely.ops import split, unary_union


class RectilinearPolygon:
    def __init__(self, polygon: Polygon):
        self.polygon = polygon
        self.constructed_lines = []
        self.min_partition_length = float("inf")
        self.best_partition = []

    def is_rectilinear(self) -> bool:
        coords = list(self.polygon.exterior.coords)
        for i in range(len(coords) - 1):
            x1, y1 = coords[i]
            x2, y2 = coords[i + 1]
            if not (x1 == x2 or y1 == y2):
                return False
        return True

    def find_convex_points(self):
        """
        Finds the convex points in the given rectilinear polygon.

        Returns:
            list: List of Points representing the convex points of the polygon.

        >>> polygon = Polygon([(0, 0), (0, 2), (0, 4), (2, 4), (2, 2), (2, 0)])
        >>> rect_polygon = RectilinearPolygon(polygon)
        >>> convex_points = rect_polygon.find_convex_points()
        >>> convex_points == [Point(0, 0), Point(0, 4), Point(2, 4), Point(2, 0)]
        True
        """
        convex_points = []
        coords = list(
            self.polygon.exterior.coords[:-1]
        )  # Exclude the last point which is the same as the first

        for i in range(len(coords)):
            x1, y1 = coords[i]
            x2, y2 = coords[
                (i + 1) % len(coords)
            ]  # Get the next point, wrapping around to the first point at the end
            x3, y3 = coords[(i + 2) % len(coords)]  # Get the point after the next one

            # Calculate the cross product of vectors (x2 - x1, y2 - y1) and (x3 - x2, y3 - y2)
            cross_product = (x2 - x1) * (y3 - y2) - (y2 - y1) * (x3 - x2)

            # Check if the cross product is positive (indicating a left turn)
            if cross_product > 0:
                convex_points.append(Point(x2, y2))

        return convex_points

    def get_grid_points(self):  # works
        """
        Returns the grid points induced by the boundary of the polygon, including the polygon vertices.
        """
        # Get the polygon vertices
        polygon_vertices = list(self.polygon.exterior.coords)

        # Get the horizontal and vertical lines formed by extending the edges of the polygon
        x_coords = set(coord[0] for coord in polygon_vertices)
        y_coords = set(coord[1] for coord in polygon_vertices)

        vertical_lines = [
            LineString([(x, min(y_coords)), (x, max(y_coords))]) for x in x_coords
        ]
        horizontal_lines = [
            LineString([(min(x_coords), y), (max(x_coords), y)]) for y in y_coords
        ]

        # Find the intersection points of the horizontal and vertical lines
        intersections = unary_union(vertical_lines + horizontal_lines).intersection(
            self.polygon
        )

        # Convert the intersection points to a set of Point objects to remove duplicates
        grid_points = set(Point(geom.coords[0]) for geom in intersections.geoms)

        # Add the polygon vertices to the grid points
        grid_points.update(map(Point, polygon_vertices))

        return list(grid_points)

    def find_matching_point(self, candidate: Point):  # works
        """
        Finds matching points on the grid inside the polygon and kitty-corner to the candidate point
        within a blocked rectangle inside the polygon.

        Args:
            candidate (Point): The candidate point.

        Returns:
            list: List of Points representing the matching points.
        """
        matching_points = []
        grid_points = self.get_grid_points()

        for point in grid_points:
            if point != candidate:  # Exclude candidate point
                # Check if the point is kitty-corner to the candidate within a blocked rectangle
                min_x = min(candidate.x, point.x)
                max_x = max(candidate.x, point.x)
                min_y = min(candidate.y, point.y)
                max_y = max(candidate.y, point.y)
                blocked_rect = Polygon(
                    [(min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y)]
                )

                if blocked_rect.within(self.polygon):
                    # Check if the line segment between candidate and point crosses the polygon exterior
                    segment = LineString([candidate, point])
                    if not any(
                        segment.crosses(LineString(self.polygon.exterior))
                        for line in self.constructed_lines
                    ):
                        matching_points.append(point)

        return matching_points

    def find_blocked_rectangle(self, candidate: Point, matching: Point):
        # Create the four edges of the potential blocked rectangle
        edge1 = LineString([candidate, Point(candidate.x, matching.y)])
        edge2 = LineString([candidate, Point(matching.x, candidate.y)])
        edge3 = LineString([Point(candidate.x, matching.y), matching])
        edge4 = LineString([Point(matching.x, candidate.y), matching])

        # Collect all edges
        all_edges = [edge1, edge2, edge3, edge4]

        # Find the segments of each edge that are not part of the polygon boundary
        internal_edges = []
        for edge in all_edges:
            difference = edge.difference(self.polygon.boundary)
            if not difference.is_empty:
                if difference.geom_type == "MultiLineString":
                    for line in difference:
                        internal_edges.append(line)
                else:
                    internal_edges.append(difference)
        # return only the new internal edges that are not part of the polygon boundary
        return internal_edges if internal_edges else None

    def split_polygon(self, lines):
        """
        Splits the polygon using the given lines.

        Args:
            lines (list[LineString]): A list of LineString objects to split the polygon.

        Returns:
            list: List of Polygon objects resulting from the split operation.

        Note: a list that contain more than one polygon is a GeometryCollection type - it can't be split using the 'split' function
        """
        if not lines:
            return [self.polygon]

        def split_geometry(geom, lines):
            polygons = []
            if isinstance(geom, Polygon):
                for line in lines:
                    split_result = split(geom, line)
                    if isinstance(split_result, GeometryCollection): # see the note above
                        polygons.extend(split_geometry(split_result, []))
                    elif isinstance(split_result, Polygon):
                        polygons.append(split_result)
            elif isinstance(geom, GeometryCollection):
                for sub_geom in geom.geoms:
                    polygons.extend(split_geometry(sub_geom, lines))
            return polygons

        polygons = split_geometry(self.polygon, lines)
        return polygons

    def find_candidate_points(self, constructed_lines: list[LineString]):
        if len(constructed_lines) == 1:
            return [
                Point(constructed_lines[0].coords[0]),
                Point(constructed_lines[0].coords[1]),
            ]
        elif len(constructed_lines) == 2:
            line1 = constructed_lines[0]
            line2 = constructed_lines[1]
            common_point = line1.intersection(line2)
            if common_point.is_empty:
                return []
            else:
                return [common_point]

    def is_rectangle(self, poly: Polygon) -> bool:
        """
        Check if a given polygon is a rectangle.
        source: https://stackoverflow.com/questions/62467829/python-check-if-shapely-polygon-is-a-rectangle
        Args:
            poly (Polygon): The polygon to be checked.

        Returns:
            bool: True if the polygon is a rectangle, False otherwise.
        """
        return poly.area == poly.minimum_rotated_rectangle.area

    def is_partitioned_into_rectangles(
        self, partitions: list[LineString]
    ) -> bool:  
        """
        Checks if the polygon is partitioned into rectangles.

        Args:
            partitions (list[LineString]): A list of LineString objects representing the partitions.

        Returns:
            bool: True if the polygon is partitioned into rectangles, False otherwise.

        >>> polygon = Polygon([(0, 0), (0, 6), (6, 6), (6, 0)])
        >>> partitions = [LineString([(0, 3), (6, 3)]), LineString([(3, 0), (3, 6)])]
        >>> rect_polygon = RectilinearPolygon(polygon)
        >>> rect_polygon.is_partitioned_into_rectangles(partitions)
        True


        >>> polygon = Polygon([(0, 0), (0, 4), (4, 4), (4, 0)])
        >>> partitions = [LineString([(0, 2), (4, 3)])]  # Non-rectangular partition
        >>> rect_polygon = RectilinearPolygon(polygon)
        >>> rect_polygon.is_partitioned_into_rectangles(partitions)
        False

        """
        if not partitions:
            return self.is_rectangle(self.polygon)
        polygons = self.split_polygon(partitions)
        return all(self.is_rectangle(poly) for poly in polygons)

    def partition(self):
        if not self.is_rectilinear():
            return None

        initial_convex_points = self.find_convex_points()
        self.recursive_partition(initial_convex_points, [])

        return self.best_partition

    def recursive_partition(  # need to add logging here
        self, candidate_points, partitionList: list[LineString]
    ):
        current_length = sum(line.length for line in partitionList)
        if self.is_partitioned_into_rectangles(partitionList):
            if current_length < self.min_partition_length:
                self.min_partition_length = current_length
                self.best_partition = partitionList
            return
        else:
            if current_length >= self.min_partition_length:  # cut this branch
                return

        for candidate in candidate_points:
            matching_points = self.find_matching_point(candidate)
            if not matching_points:
                continue

            for matching_point in matching_points:
                new_lines = self.find_blocked_rectangle(candidate, matching_point)
                if new_lines is None:
                    continue

                # new_polygons = self.split_polygon(new_lines)
                new_candidate_points = self.find_candidate_points(new_lines)

                self.recursive_partition(
                    new_candidate_points, partitionList + new_lines
                )


if __name__ == "__main__":
    # Create the polygon instance
    polygon = Polygon([(2, 0), (6, 0), (6, 4), (8, 4), (8, 6), (0, 6), (0, 4), (2, 4)])

    # Create a RectilinearPolygon instance
    rectilinear_polygon = RectilinearPolygon(polygon)

    # Get the partition result
    partition_result = rectilinear_polygon.partition()

    if partition_result:
        # Process the partition result
        print("Partition result:", partition_result)
    else:
        print("Partition could not be found.")
