"""
ECE 462/662 - HW1 Part 2, Problem 1: Reflex Agent Implementation
Self-Driving Car Security Patrolling

This module implements a simple reflex agent for a self-driving car
that monitors a neighborhood for security threats.
"""

def self_driving_car(intruder_detected=False, suspicious_behavior=False, fire_detected=False, fire_severity=0.0, target_location=None):
    """
    Reflex agent for security patrolling self-driving car.
    
    Args:
        intruder_detected (bool): Whether an intruder is detected
        suspicious_behavior (bool): Whether suspicious behavior is observed
        fire_detected (bool): Whether fire is detected by smoke sensor
        fire_severity (float): Fire severity on scale 0-10 (threshold = 7.0)
        target_location (str): Location of emergency (if applicable)
    
    Returns:
        str: Description of action taken by the agent
    """
    
    # Rule 1: Intruder takes highest priority
    if intruder_detected:
        return (f"EMERGENCY: Intruder detected! Emitting alarm. "
                f"Moving to {target_location} at maximum speed (25 m/s). "
                f"Securing the area.")
    
    # Rule 2: Suspicious behavior is second priority
    if suspicious_behavior:
        return (f"ALERT: Suspicious behavior detected at {target_location}. "
                f"Emitting warning alarm. Increasing speed to 20 m/s. "
                f"Sending alert to control room.")
    
    # Rule 3: Fire detection with severity threshold
    if fire_detected and fire_severity >= 7.0:
        return (f"CRITICAL: Fire detected (severity: {fire_severity}/10). "
                f"Calling fire department immediately. "
                f"Moving to fire location at high speed (20 m/s).")
    
    if fire_detected and fire_severity < 7.0:
        return (f"WARNING: Fire detected (severity: {fire_severity}/10). "
                f"Monitoring situation. Continuing patrol at normal speed.")
    
    # Rule 4: No emergency detected - continue normal patrol
    return ("All systems normal. Continuing neighborhood patrol at 10 m/s.")


def display_instructions():
    """Display usage instructions for the system."""
    print("=" * 60)
    print("   SELF-DRIVING CAR SECURITY AGENT - INTERACTIVE MODE")
    print("=" * 60)
    print()
    print("INSTRUCTIONS:")
    print("-" * 60)
    print("Enter emergency types using the following codes:")
    print()
    print("  A  - Intruder detected (highest priority)")
    print("  B  - Suspicious behavior observed")
    print("  F  - Fire detected (will prompt for severity 0-10)")
    print()
    print("For MULTIPLE EMERGENCIES, enter codes separated by commas.")
    print("  Example: A,F  (intruder + fire)")
    print("  Example: B,F  (suspicious behavior + fire)")
    print("  Example: A,B,F (all emergencies)")
    print()
    print("Press ENTER with no input for normal patrol (no emergency).")
    print("Type 'quit' or 'q' to exit the program.")
    print("-" * 60)
    print()


def get_user_input():
    """Get and parse user input for emergency types."""
    while True:
        display_instructions()

        emergency_input = input("Enter emergency code(s) [A/B/F or comma-separated]: ").strip().upper()

        if emergency_input.lower() in ['quit', 'q']:
            print("Exiting security agent. Goodbye!")
            return None

        # Parse input
        intruder = False
        suspicious = False
        fire = False
        fire_severity = 0.0
        target_location = None

        if emergency_input == "":
            # No emergency - normal patrol
            pass
        else:
            # Parse comma-separated codes
            codes = [c.strip() for c in emergency_input.split(',')]

            for code in codes:
                if code == 'A':
                    intruder = True
                elif code == 'B':
                    suspicious = True
                elif code == 'F':
                    fire = True
                elif code not in ['A', 'B', 'F', '']:
                    print(f"\nInvalid code: '{code}'. Please use A, B, or F.\n")
                    input("Press ENTER to try again...")
                    continue

        # Get fire severity if fire detected
        if fire:
            while True:
                try:
                    severity_input = input("Enter fire severity (0-10): ").strip()
                    fire_severity = float(severity_input)
                    if 0 <= fire_severity <= 10:
                        break
                    else:
                        print("Severity must be between 0 and 10.")
                except ValueError:
                    print("Please enter a valid number.")

        # Get target location if any emergency detected
        if intruder or suspicious or fire:
            target_location = input("Enter target location (e.g., 'Building A'): ").strip()
            if not target_location:
                target_location = "Unknown Location"

        # Call the agent and display result
        print("\n" + "=" * 60)
        print("AGENT RESPONSE:")
        print("=" * 60)
        result = self_driving_car(
            intruder_detected=intruder,
            suspicious_behavior=suspicious,
            fire_detected=fire,
            fire_severity=fire_severity,
            target_location=target_location
        )
        print(result)
        print("=" * 60 + "\n")

        input("Press ENTER to continue...")
        print("\n")


def run_test_cases():
    """Run predefined test cases."""
    print("=== Self-Driving Car Security Agent - Test Cases ===\n")

    # Test 1: Intruder detected (highest priority)
    print("Test 1 - Intruder Detected:")
    result = self_driving_car(intruder_detected=True, target_location="Building A")
    print(result)
    print()

    # Test 2: Suspicious behavior
    print("Test 2 - Suspicious Behavior:")
    result = self_driving_car(suspicious_behavior=True, target_location="Corner of 5th Ave")
    print(result)
    print()

    # Test 3: Fire with high severity
    print("Test 3 - Fire (High Severity):")
    result = self_driving_car(fire_detected=True, fire_severity=8.5)
    print(result)
    print()

    # Test 4: Fire with low severity
    print("Test 4 - Fire (Low Severity):")
    result = self_driving_car(fire_detected=True, fire_severity=5.0)
    print(result)
    print()

    # Test 5: Multiple emergencies (prioritize intruder > suspicious > fire)
    print("Test 5 - Multiple Emergencies (Intruder + Fire):")
    result = self_driving_car(intruder_detected=True, fire_detected=True,
                             fire_severity=9.0, target_location="Building C")
    print(result)
    print()

    # Test 6: Multiple emergencies (Suspicious + Fire)
    print("Test 6 - Multiple Emergencies (Suspicious + Fire):")
    result = self_driving_car(suspicious_behavior=True, fire_detected=True,
                             fire_severity=8.0, target_location="Park Area")
    print(result)
    print()

    # Test 7: No emergency
    print("Test 7 - No Emergency:")
    result = self_driving_car()
    print(result)
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("   SELF-DRIVING CAR SECURITY PATROL SYSTEM")
    print("=" * 60)
    print()
    print("Select mode:")
    print("  1 - Interactive mode (enter your own emergencies)")
    print("  2 - Run predefined test cases")
    print("  q - Quit")
    print()

    mode = input("Enter choice [1/2/q]: ").strip().lower()

    if mode == '1':
        get_user_input()
    elif mode == '2':
        run_test_cases()
    elif mode in ['q', 'quit']:
        print("Goodbye!")
    else:
        print("Invalid choice. Running test cases by default.\n")
        run_test_cases()


# ============================================================================
# ANALYSIS - Answers to follow-up questions
# ============================================================================

"""
A. Is it a good design to formulate self-driving cars as simple reflex agents? Why?

NO - Simple reflex agents are NOT adequate for self-driving cars. While effective
for simple patrol tasks with clear, immediate rules, they lack:

1. STATE MEMORY: A reflex agent cannot remember where it has patrolled, making
   coverage incomplete. It cannot maintain a model of visited locations.

2. TEMPORAL REASONING: A fire might need sustained monitoring, not just immediate
   action. The agent cannot track mission history (e.g., "Have I already helped
   this location?").

3. COMPLEX PLANNING: Multi-step responses (e.g., "secure the area, then return to
   patrol") require planning beyond immediate percept → action mapping.

4. SENSOR UNCERTAINTY: Real sensors have noise. A reflex agent cannot reason about
   confidence levels or cross-validate sensor readings.

5. DYNAMIC PRIORITIES: The agent cannot adjust priorities based on context (e.g.,
   if an intruder is already contained, suspicious behavior might become priority).

---

B. What are the potential points of failure of this design? How can we improve it?

FAILURE POINTS:

1. FALSE ALARMS: A single sensor spike triggers full emergency response, even if
   the alert is spurious. No confirmation mechanism.
   → FIX: Add sensor fusion (require multiple sensors to confirm before acting).

2. INFINITE LOOPS: If a fire is detected at location X, the agent moves there
   but has no memory. If it loops back through the area, it reacts identically
   again, potentially in conflict with ongoing operations.
   → FIX: Maintain visited state / mark handled locations.

3. MISSED COVERAGE: Without memory, the agent might patrol inefficiently, missing
   entire sections of the neighborhood.
   → FIX: Use a model-based agent with a spatial map / patrol schedule.

4. CONFLICTING ACTIONS: If both fire and intruder are present but the agent
   can only respond to one, there's no recovery mechanism to handle the other.
   → FIX: Add goal-based planning to create action sequences, or use a queue.

5. NO LEARNING: The agent cannot improve response times or patterns over time.
   → FIX: Add a learning component (e.g., reinforce frequently effective patrol routes).

6. SENSOR LATENCY: Real sensors have delays. The agent might act on outdated
   information, causing safety issues.
   → FIX: Use a model-based agent with prediction.

IMPROVEMENTS:

→ Model-Based Agent: Track internal map of neighborhood, mark hazards/visited
  locations, improve patrol coverage.

→ Goal-Based Agent: Set explicit goals (e.g., "patrol all buildings," "respond
  to all emergencies") and plan action sequences.

→ Utility-Based Agent: Weigh multiple objectives (e.g., response speed vs.
  coverage completeness) to make trade-off decisions.

---

C. What other agent design would you prefer instead of this one? Why?

PREFERRED DESIGN: MODEL-BASED + GOAL-BASED + UTILITY-BASED HYBRID

A hierarchical agent architecture:

LAYER 1 - REACTIVE (Reflex-like):
  - Real-time sensor processing for immediate threats (emergency response).
  - If fire severity > 9.0 or intruder confirmed, respond immediately.

LAYER 2 - DELIBERATIVE (Model-Based):
  - Maintain an internal map of the neighborhood with:
    * Visited locations
    * Known hazards (fire zones, construction areas)
    * Patrol coverage statistics
  - Update the model as the agent explores.
  - Use stored knowledge to avoid re-triggering at same locations.

LAYER 3 - GOAL-BASED:
  - Long-term goals: "Cover all buildings," "Minimize emergency response time,"
    "Maintain 24-hour patrol of high-risk areas."
  - Plan sequences of actions: ["Go to Building A", "Check perimeter", 
    "Return to patrol base"]

LAYER 4 - LEARNING:
  - Track which patrol patterns are most effective.
  - Learn typical emergency patterns (e.g., fires more common near workshops).
  - Adapt patrol intensity based on historical data.

REASONING:
This design combines speed (reactive layer for emergencies) with intelligence
(deliberative planning, learning). A simple reflex agent is insufficient for
autonomous vehicles, which must be safe, efficient, and adaptive.
"""