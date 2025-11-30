#!/usr/bin/env python
"""Test script for G-FOLD guidance in isolation.

This script tests the G-FOLD solver with various scenarios to understand
its performance characteristics and identify issues.
"""

import time
import numpy as np
from rocket.dynamics.state import State
from flight.guidance.gfold import GFOLDGuidance


def test_simple_vertical_landing():
    """Test simple vertical descent - should be easy case."""
    print("=" * 60)
    print("TEST 1: Simple Vertical Landing")
    print("=" * 60)
    
    # Landing site at origin for simplicity
    landing_site = np.array([6378137.0, 0.0, 0.0])  # On equator
    
    # Start 5 km directly above, descending at 200 m/s
    position = landing_site + np.array([5000.0, 0.0, 0.0])  # 5 km up
    velocity = np.array([-200.0, 0.0, 0.0])  # Descending
    
    state = State(
        position=position,
        velocity=velocity,
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_velocity=np.zeros(3),
        mass=30000.0,
        time=0.0
    )
    
    gfold = GFOLDGuidance(
        target_position=landing_site,
        max_thrust=1000000.0,  # 1 MN
        min_thrust=100000.0,   # 100 kN
        dry_mass=20000.0,
    )
    
    print(f"Start altitude: 5 km")
    print(f"Start velocity: -200 m/s (descending)")
    print(f"Mass: 30,000 kg")
    print(f"Max thrust: 1 MN")
    print()
    
    start = time.time()
    result = gfold.solve(state)
    elapsed = time.time() - start
    
    if result:
        print(f"SUCCESS in {elapsed:.2f}s")
        print(f"  Burn duration: {result['tf']:.1f} s")
        print(f"  Final pos error: {np.linalg.norm(result['position'][-1] - landing_site):.1f} m")
        print(f"  Final velocity: {np.linalg.norm(result['velocity'][-1]):.2f} m/s")
        print(f"  Fuel used: {result['mass'][0] - result['mass'][-1]:.0f} kg")
    else:
        print(f"FAILED in {elapsed:.2f}s")
    
    return result is not None


def test_angled_approach():
    """Test approach with horizontal velocity component."""
    print("\n" + "=" * 60)
    print("TEST 2: Angled Approach")
    print("=" * 60)
    
    landing_site = np.array([6378137.0, 0.0, 0.0])
    
    # Start 10 km up and 5 km downrange, moving toward site
    position = landing_site + np.array([10000.0, 5000.0, 0.0])
    velocity = np.array([-300.0, -200.0, 0.0])  # Descending and moving toward site
    
    state = State(
        position=position,
        velocity=velocity,
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_velocity=np.zeros(3),
        mass=40000.0,
        time=0.0
    )
    
    gfold = GFOLDGuidance(
        target_position=landing_site,
        max_thrust=2000000.0,
        min_thrust=200000.0,
        dry_mass=20000.0,
    )
    
    print(f"Start: 10 km up, 5 km downrange")
    print(f"Velocity: descending + horizontal")
    print()
    
    start = time.time()
    result = gfold.solve(state)
    elapsed = time.time() - start
    
    if result:
        print(f"SUCCESS in {elapsed:.2f}s")
        print(f"  Burn duration: {result['tf']:.1f} s")
        print(f"  Final pos error: {np.linalg.norm(result['position'][-1] - landing_site):.1f} m")
        print(f"  Final velocity: {np.linalg.norm(result['velocity'][-1]):.2f} m/s")
    else:
        print(f"FAILED in {elapsed:.2f}s")
    
    return result is not None


def test_rtls_scenario():
    """Test realistic RTLS scenario - far from landing site."""
    print("\n" + "=" * 60)
    print("TEST 3: RTLS Scenario (Challenging)")
    print("=" * 60)
    
    # Cape Canaveral landing site
    landing_site = np.array([915477.28738129, -5529950.00866849, 3043383.94368594])
    
    # Realistic RTLS state: 20 km altitude, 50 km from site, high speed
    r_hat = landing_site / np.linalg.norm(landing_site)
    # Position: 20 km up, offset horizontally
    offset = np.array([50000.0, 0.0, 0.0])  # 50 km horizontal offset
    offset = offset - np.dot(offset, r_hat) * r_hat  # Make it perpendicular to radial
    position = landing_site + r_hat * 20000 + offset
    
    # Velocity toward landing site
    to_site = landing_site - position
    to_site_hat = to_site / np.linalg.norm(to_site)
    velocity = to_site_hat * 500 + r_hat * (-300)  # 500 m/s toward site, 300 m/s down
    
    state = State(
        position=position,
        velocity=velocity,
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_velocity=np.zeros(3),
        mass=50000.0,
        time=0.0
    )
    
    gfold = GFOLDGuidance(
        target_position=landing_site,
        max_thrust=4225000.0,  # 5 Merlins
        min_thrust=338000.0,
        dry_mass=25000.0,
    )
    
    # Compute actual distances
    rel_pos = position - landing_site
    alt = np.dot(rel_pos, r_hat)
    h_dist = np.linalg.norm(rel_pos - alt * r_hat)
    
    print(f"Altitude above site: {alt/1000:.1f} km")
    print(f"Horizontal distance: {h_dist/1000:.1f} km")
    print(f"Speed: {np.linalg.norm(velocity):.0f} m/s")
    print()
    
    start = time.time()
    result = gfold.solve(state)
    elapsed = time.time() - start
    
    if result:
        print(f"SUCCESS in {elapsed:.2f}s")
        print(f"  Burn duration: {result['tf']:.1f} s")
        print(f"  Final pos error: {np.linalg.norm(result['position'][-1] - landing_site):.1f} m")
        print(f"  Final velocity: {np.linalg.norm(result['velocity'][-1]):.2f} m/s")
        print(f"  Fuel used: {result['mass'][0] - result['mass'][-1]:.0f} kg")
    else:
        print(f"FAILED in {elapsed:.2f}s")
        print("  Trying with relaxed constraints...")
        
        # Try with more relaxed constraints
        gfold2 = GFOLDGuidance(
            target_position=landing_site,
            max_thrust=4225000.0,
            min_thrust=100000.0,  # Lower min
            dry_mass=25000.0,
            glideslope_tan=10.0,  # Very relaxed
            max_tilt=np.radians(85.0),
            n_nodes=20,  # Fewer nodes
        )
        
        start = time.time()
        result2 = gfold2.solve(state)
        elapsed = time.time() - start
        
        if result2:
            print(f"  SUCCESS with relaxed constraints in {elapsed:.2f}s")
        else:
            print(f"  Still FAILED")
    
    return result is not None


def test_solver_timing():
    """Test solver timing with different node counts."""
    print("\n" + "=" * 60)
    print("TEST 4: Solver Timing")
    print("=" * 60)
    
    landing_site = np.array([6378137.0, 0.0, 0.0])
    position = landing_site + np.array([10000.0, 0.0, 0.0])
    velocity = np.array([-200.0, 0.0, 0.0])
    
    state = State(
        position=position,
        velocity=velocity,
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_velocity=np.zeros(3),
        mass=30000.0,
        time=0.0
    )
    
    for n_nodes in [10, 15, 20, 25, 30, 40, 50]:
        gfold = GFOLDGuidance(
            target_position=landing_site,
            max_thrust=1000000.0,
            min_thrust=100000.0,
            dry_mass=20000.0,
            n_nodes=n_nodes,
        )
        
        start = time.time()
        result = gfold.solve(state)
        elapsed = time.time() - start
        
        status = "OK" if result else "FAIL"
        print(f"  n_nodes={n_nodes:2d}: {elapsed:.3f}s [{status}]")


def test_get_command_caching():
    """Test that get_command properly caches solutions."""
    print("\n" + "=" * 60)
    print("TEST 5: Command Caching")
    print("=" * 60)
    
    landing_site = np.array([6378137.0, 0.0, 0.0])
    position = landing_site + np.array([5000.0, 0.0, 0.0])
    velocity = np.array([-100.0, 0.0, 0.0])
    
    gfold = GFOLDGuidance(
        target_position=landing_site,
        max_thrust=1000000.0,
        min_thrust=100000.0,
        dry_mass=20000.0,
        n_nodes=15,
    )
    
    # First call - should solve
    state = State(
        position=position,
        velocity=velocity,
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_velocity=np.zeros(3),
        mass=30000.0,
        time=0.0
    )
    
    start = time.time()
    cmd1 = gfold.get_command(state)
    t1 = time.time() - start
    print(f"First call: {t1:.3f}s - {cmd1['phase']}")
    
    # Second call with same time - should use cache
    start = time.time()
    cmd2 = gfold.get_command(state)
    t2 = time.time() - start
    print(f"Second call (cached): {t2:.3f}s - {cmd2['phase']}")
    
    # Call with slightly later time - should still use cache
    state2 = State(
        position=position + velocity * 0.5,
        velocity=velocity,
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_velocity=np.zeros(3),
        mass=30000.0,
        time=0.5
    )
    
    start = time.time()
    cmd3 = gfold.get_command(state2)
    t3 = time.time() - start
    print(f"Third call (t+0.5s): {t3:.3f}s - {cmd3['phase']}")
    
    # Call with time > 5s later - should re-solve
    state3 = State(
        position=position + velocity * 6,
        velocity=velocity,
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_velocity=np.zeros(3),
        mass=30000.0,
        time=6.0
    )
    
    start = time.time()
    cmd4 = gfold.get_command(state3)
    t4 = time.time() - start
    print(f"Fourth call (t+6s, re-solve): {t4:.3f}s - {cmd4['phase']}")


if __name__ == "__main__":
    print("G-FOLD Guidance Test Suite")
    print("=" * 60)
    print()
    
    results = []
    
    results.append(("Simple Vertical", test_simple_vertical_landing()))
    results.append(("Angled Approach", test_angled_approach()))
    results.append(("RTLS Scenario", test_rtls_scenario()))
    
    test_solver_timing()
    test_get_command_caching()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")


