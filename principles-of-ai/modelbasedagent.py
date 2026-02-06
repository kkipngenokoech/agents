"""
ECE 462/662 - HW1 Part 2, Problem 3: Model-Based Agent
Exploration of Martian Lava Tubes

This module implements a model-based agent that maintains an internal map
of a partially observable environment and uses it to safely navigate.

The rover uses a persistent internal model to track:
  - Explored vs unexplored cells
  - Safe floor cells
  - Hazardous lava cells
  
This allows the agent to learn from past observations and avoid
previously discovered lava hazards.
"""

import numpy as np
from collections import deque
from typing import Tuple, Optional


class ModelBasedRover:
    """
    A model-based agent for exploring Martian lava tubes.
    
    Maintains an internal 2D map that persists across environment resets,
    learning from observations to navigate safely.
    """
    
    def __init__(self, grid_size: int = 50):
        """
        Initialize the model-based rover.
        
        Args:
            grid_size: Size of the internal map (grid_size x grid_size)
        """
        self.grid_size = grid_size
        self.position = (grid_size // 2, grid_size // 2)  # Start in center
        self.heading = 0  # 0=North, 1=East, 2=South, 3=West
        
        # Internal model: 0=unexplored, 1=safe, -1=lava
        self.model = np.zeros((grid_size, grid_size), dtype=int)
        
        # Mark starting position as safe
        self.model[self.position] = 1
        
        self.visited_count = 0
        self.goal_reached = False
        
    def ego_to_global_coordinates(self, ego_pos: Tuple[int, int]) -> Tuple[int, int]:
        """
        Convert ego-centric (relative) coordinates to global coordinates.
        
        The environment provides a 7x7 observation relative to the rover's
        current position and heading. This function translates that to global
        map coordinates.
        
        Args:
            ego_pos: Position in ego-centric frame (row, col) where
                    (3, 3) is the rover's current position
        
        Returns:
            Global (x, y) coordinates on the rover's internal map
        
        COORDINATE MAPPING LOGIC:
        - Ego frame: (3, 3) = rover position, (0, 0) = forward-left corner
        - Global frame: Absolute position on the rover's persistent map
        
        The mapping depends on the rover's current heading:
          Heading 0 (North): Ego forward = North on map
          Heading 1 (East):  Ego forward = East on map
          Heading 2 (South): Ego forward = South on map
          Heading 3 (West):  Ego forward = West on map
        """
        ego_row, ego_col = ego_pos
        
        # Offset from rover center in ego frame
        offset_row = ego_row - 3  # -3 to +3
        offset_col = ego_col - 3  # -3 to +3
        
        # Rotate offset based on heading
        if self.heading == 0:  # North (no rotation)
            global_row = self.position[0] - offset_row
            global_col = self.position[1] + offset_col
        elif self.heading == 1:  # East (90° clockwise)
            global_row = self.position[0] + offset_col
            global_col = self.position[1] + offset_row
        elif self.heading == 2:  # South (180°)
            global_row = self.position[0] + offset_row
            global_col = self.position[1] - offset_col
        elif self.heading == 3:  # West (270° clockwise)
            global_row = self.position[0] - offset_col
            global_col = self.position[1] - offset_row
        
        return (global_row, global_col)
    
    def update_model(self, observation: np.ndarray):
        """
        Update the internal model based on current observation.
        
        Args:
            observation: 7x7 grid from the environment where:
                        0 = empty/safe, 10 = lava
        """
        for ego_row in range(7):
            for ego_col in range(7):
                global_pos = self.ego_to_global_coordinates((ego_row, ego_col))
                
                # Bounds check
                if (0 <= global_pos[0] < self.grid_size and 
                    0 <= global_pos[1] < self.grid_size):
                    
                    cell_value = observation[ego_row, ego_col]
                    
                    if cell_value == 10:  # Lava detected
                        self.model[global_pos] = -1  # Mark as hazard
                    elif cell_value == 0:  # Safe floor
                        self.model[global_pos] = 1   # Mark as safe
    
    def find_unexplored_target(self) -> Optional[Tuple[int, int]]:
        """
        Find the nearest unexplored cell using BFS.
        
        Returns:
            Coordinates of nearest unexplored cell, or None if all explored
        """
        visited = set()
        queue = deque([self.position])
        visited.add(self.position)
        
        while queue:
            current = queue.popleft()
            
            # Check if this cell is unexplored
            if self.model[current] == 0:
                return current
            
            # Explore neighbors
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (current[0] + dr, current[1] + dc)
                
                if (0 <= neighbor[0] < self.grid_size and
                    0 <= neighbor[1] < self.grid_size and
                    neighbor not in visited and
                    self.model[neighbor] != -1):  # Don't revisit lava
                    
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return None  # All cells explored or blocked
    
    def find_path_to_target(self, target: Tuple[int, int]) -> Optional[list]:
        """
        Find a safe path to target using BFS (avoiding lava).
        
        Args:
            target: Target coordinates
        
        Returns:
            List of coordinates from current position to target
        """
        queue = deque([(self.position, [self.position])])
        visited = {self.position}
        
        while queue:
            current, path = queue.popleft()
            
            if current == target:
                return path
            
            # Check 4-connected neighbors
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (current[0] + dr, current[1] + dc)
                
                if (0 <= neighbor[0] < self.grid_size and
                    0 <= neighbor[1] < self.grid_size and
                    neighbor not in visited and
                    self.model[neighbor] != -1):  # Avoid lava
                    
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None  # No path found
    
    def move_towards(self, target: Tuple[int, int]) -> str:
        """
        Move one step towards target.
        
        Args:
            target: Target coordinates
        
        Returns:
            Description of action taken
        """
        path = self.find_path_to_target(target)
        
        if path is None or len(path) <= 1:
            return "Cannot reach target (blocked by lava)"
        
        next_pos = path[1]  # Next step along path
        self.position = next_pos
        self.visited_count += 1
        
        direction = {
            (-1, 0): "North",
            (1, 0): "South",
            (0, -1): "West",
            (0, 1): "East"
        }
        
        move = (next_pos[0] - self.position[0], 
                next_pos[1] - self.position[1])
        dir_name = direction.get(move, "Unknown")
        
        return f"Moved {dir_name} to {next_pos}"
    
    def select_action(self, observation: np.ndarray) -> str:
        """
        Decide next action based on current observation and internal model.
        
        Args:
            observation: 7x7 current observation from environment
        
        Returns:
            Description of action to take
        """
        # Step 1: Update internal model with new observations
        self.update_model(observation)
        
        # Step 2: Find nearest unexplored cell
        target = self.find_unexplored_target()
        
        if target is None:
            return "Exploration complete - all cells explored"
        
        # Step 3: Move towards unexplored cell
        distance = abs(target[0] - self.position[0]) + abs(target[1] - self.position[1])
        
        return f"Moving toward unexplored cell at {target} (distance: {distance})"
    
    def get_map_visualization(self) -> str:
        """
        Return ASCII visualization of the internal map.
        
        Returns:
            String representation of the explored regions
        """
        visual = []
        for i in range(self.grid_size):
            row = ""
            for j in range(self.grid_size):
                if (i, j) == self.position:
                    row += "R"  # Rover
                elif self.model[i, j] == -1:
                    row += "#"  # Lava (hazard)
                elif self.model[i, j] == 1:
                    row += "."  # Safe floor
                else:
                    row += " "  # Unexplored
            visual.append(row)
        
        return "\n".join(visual)


# ============================================================================
# PEAS DESCRIPTION
# ============================================================================

PEAS_DESCRIPTION = """
PEAS DESCRIPTION: Exploration of Martian Lava Tubes

PERFORMANCE MEASURE:
  - Primary: Safely reach the goal location (exit of lava tube)
  - Secondary: Maximize map coverage (explore as many cells as possible)
  - Penalty: Enter lava cell (mission fails, environment resets)
  - Bonus: Reach goal multiple times using accumulated knowledge

ENVIRONMENT:
  - Martian lava tube (underground cavern)
  - 7x7 grid cells visible to rover sensors at any moment
  - Partially Observable: Rover only sees local 7x7 observation
  - Goal location: Fixed location the rover must reach
  - Lava hazards: Random placement, fatal if touched
  - Bedrock/safe floor: Walkable terrain

ACTUATORS:
  - Movement: 4-directional movement (North, South, East, West)
  - One step per action (each cell = one unit)
  - Rotation: Can change heading (0=North, 1=East, 2=South, 3=West)

SENSORS:
  - Vision: 7x7 grid observation (ego-centric view)
  - Cell detection: Identifies lava (value=10) vs safe floor (value=0)
  - Position awareness: Knows current (x, y) on local 7x7 grid
  - Heading awareness: Knows current compass direction
  - Localization: Can track position relative to goal

ENVIRONMENT PROPERTIES:
  - Partially Observable: Only 7x7 local view available
  - Single Agent: Only the rover, no competing agents
  - Stochastic: Lava placement may be random/adversarial
  - Sequential: Actions accumulate, history matters
  - Dynamic: Lava locations fixed once revealed
  - Continuous: Grid cells are discrete but space is physically continuous
  - Unknown: Lava locations unknown initially

KEY CHALLENGE:
The environment is partially observable - the rover cannot see beyond its
7x7 sensor range. This requires maintaining an internal model (persistent
across resets) to avoid revisiting known lava cells. A pure reflex agent
would fail as it couldn't remember hazard locations.
"""


# ============================================================================
# ANALYSIS: EGO-CENTRIC TO GLOBAL COORDINATE TRANSFORMATION
# ============================================================================

ANALYSIS = """
EGO-CENTRIC TO GLOBAL COORDINATE TRANSFORMATION ANALYSIS

PROBLEM:
The environment provides observations as 7x7 grids in ego-centric (relative)
coordinates, where the rover is always at position (3,3). The rover must
translate these into global coordinates on its persistent internal map to
properly track hazards and safe areas across missions.

SOLUTION APPROACH:

1. RELATIVE OFFSET CALCULATION:
   The ego-centric grid has the rover at center (3, 3).
   Any cell (r, c) has offset from rover center:
   - offset_row = r - 3  (range: -3 to +3)
   - offset_col = c - 3  (range: -3 to +3)

2. HEADING-DEPENDENT ROTATION:
   The rover's heading determines how to interpret the relative coordinates:
   
   HEADING 0 (NORTH - facing north):
     - Ego forward direction = North on map
     - offset_row < 0 means "ahead" (north) on actual map
     - offset_col corresponds to east-west
     - Global = (rover_r - offset_r, rover_c + offset_c)
   
   HEADING 1 (EAST - facing east):
     - Ego forward direction = East on map
     - 90° rotation of the coordinate frame
     - Global = (rover_r + offset_c, rover_c + offset_r)
   
   HEADING 2 (SOUTH - facing south):
     - Ego forward direction = South on map
     - 180° rotation
     - Global = (rover_r + offset_r, rover_c - offset_c)
   
   HEADING 3 (WEST - facing west):
     - Ego forward direction = West on map
     - 270° rotation
     - Global = (rover_r - offset_c, rover_c - offset_r)

3. BOUNDS CHECKING:
   Global coordinates must be valid on the map (0 to grid_size-1).
   Out-of-bounds cells are ignored (already explored or don't exist).

4. PERSISTENCE ACROSS RESETS:
   Critical advantage of this approach:
   - First mission: Explores freely, builds partial map
   - Environment resets (lava hit or goal reached)
   - Second mission: Uses stored map to avoid known lava
   - Knowledge accumulates, enabling safe exploration paths
   
   Example:
   Mission 1: Rover discovers lava at global position (25, 30)
             Marks map[25][30] = -1
             
   Mission 2: Rover sees same area from different angle
             Converts ego observation to global coords
             Avoids that cell because map[25][30] == -1
             
   This creates emergent learning behavior without explicit learning.

WHY THIS WORKS:
The consistent transformation from ego to global coordinates ensures that:
1. Same physical location is always mapped to same global coordinate
2. Different approaches to same location converge on same map entry
3. Hazard knowledge persists despite environment resets
4. Agent builds increasingly accurate model over time

EXAMPLE TRANSFORMATION:
Suppose rover at global position (25, 25), heading North.
Ego observation shows lava at ego position (1, 4):
- offset = (1-3, 4-3) = (-2, 1)
- Heading North: global = (25 - (-2), 25 + 1) = (27, 26)
- Mark map[27][26] = -1

Next mission, rover at different global position (30, 28), same heading:
Ego observation shows object at (1, 4):
- Same ego position, different global position
- offset = (-2, 1)
- Heading North: global = (30 - (-2), 28 + 1) = (32, 29) ✓
- Different cell (correctly different global coords)

This demonstrates the transformation's correctness across different rover positions.
"""


# ============================================================================
# TEST AND DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    
    print("=" * 80)
    print("MODEL-BASED ROVER AGENT - MARTIAN LAVA TUBE EXPLORATION")
    print("=" * 80)
    print()
    
    print(PEAS_DESCRIPTION)
    print()
    print("=" * 80)
    print(ANALYSIS)
    print()
    
    # Initialize rover
    rover = ModelBasedRover(grid_size=30)
    
    print("=" * 80)
    print("SIMULATION: Internal Map Development")
    print("=" * 80)
    print()
    
    # Simulate first few observations
    print("Initial State:")
    print(f"  Position: {rover.position}")
    print(f"  Heading: {rover.heading} (North)")
    print(f"  Explored cells: {np.count_nonzero(rover.model)}")
    print()
    
    # Simulate receiving observations
    print("Processing observations and updating internal model...")
    print()
    
    # Create sample observation (7x7 grid with some lava)
    sample_obs = np.zeros((7, 7), dtype=int)
    sample_obs[2:4, 3:5] = 10  # Lava at offset positions
    
    print("Sample ego-centric observation (7x7):")
    print("(0 = safe, 10 = lava, . = our position at center)")
    obs_visual = np.zeros((7, 7), dtype=str)
    obs_visual[:] = "."
    for i in range(7):
        for j in range(7):
            if sample_obs[i, j] == 10:
                obs_visual[i, j] = "#"
    
    print("\n".join(["  " + "".join(row) for row in obs_visual]))
    print()
    
    rover.update_model(sample_obs)
    
    print("After updating internal model:")
    print(f"  Explored cells: {np.count_nonzero(rover.model)}")
    print(f"  Lava cells discovered: {np.count_nonzero(rover.model == -1)}")
    print()
    
    print("Partial internal map (centered on rover position):")
    print("(R = rover, # = lava, . = safe, space = unexplored)")
    print()
    
    # Show 15x15 section around rover
    center = rover.position
    start_r = max(0, center[0] - 7)
    end_r = min(rover.grid_size, center[0] + 8)
    start_c = max(0, center[1] - 7)
    end_c = min(rover.grid_size, center[1] + 8)
    
    for i in range(start_r, end_r):
        row = ""
        for j in range(start_c, end_c):
            if (i, j) == rover.position:
                row += "R"
            elif rover.model[i, j] == -1:
                row += "#"
            elif rover.model[i, j] == 1:
                row += "."
            else:
                row += " "
        print("  " + row)
    
    print()
    print("=" * 80)
    print("COORDINATE TRANSFORMATION EXAMPLE")
    print("=" * 80)
    print()
    
    # Show transformation details
    test_ego_pos = (1, 4)
    global_pos = rover.ego_to_global_coordinates(test_ego_pos)
    
    print(f"Rover position: {rover.position}")
    print(f"Rover heading: {rover.heading} (North)")
    print(f"Ego-centric observation at: {test_ego_pos}")
    print(f"  Offset from center: ({test_ego_pos[0]-3}, {test_ego_pos[1]-3}) = (-2, 1)")
    print(f"  Global coordinates: {global_pos}")
    print()
    print("This mapping allows the rover to:")
    print("  1. Correctly place observations on persistent map")
    print("  2. Recognize same location from different angles")
    print("  3. Build unified model across multiple exploration runs")
    print()