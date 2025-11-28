import React, { useState, useEffect, useMemo, useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Stars, Sphere, Line, Html } from '@react-three/drei';
import * as THREE from 'three';

// --- Types ---
interface FlightData {
  metadata: {
    target_altitude: number;
    landing_site: number[];
    staging_time: number;
  };
  stages: StageData[];
}

interface StageData {
  id: string;
  name: string;
  times: number[];
  positions: number[][];
  velocities: number[][];
  phases: number[];
}

// --- Components ---

function Earth() {
  return (
    <Sphere args={[6378.137, 64, 64]} position={[0, 0, 0]}>
      <meshStandardMaterial 
        color="#1e3f66" 
        roughness={0.8}
        metalness={0.1}
      />
    </Sphere>
  );
}

function LaunchPad({ position }: { position: number[] }) {
  // Convert ECI to roughly km for visualization scale
  const pos = new THREE.Vector3(position[0]/1000, position[1]/1000, position[2]/1000);
  
  return (
    <mesh position={pos}>
      <boxGeometry args={[2, 2, 2]} />
      <meshBasicMaterial color="#00ff00" />
      <Html distanceFactor={500}>
        <div style={{ color: '#00ff00', fontSize: '12px' }}>PAD</div>
      </Html>
    </mesh>
  );
}

function TrajectoryLine({ points, color }: { points: THREE.Vector3[], color: string }) {
  return (
    <Line
      points={points}
      color={color}
      lineWidth={2}
    />
  );
}

function Rocket({ 
  stageData, 
  time, 
  color 
}: { 
  stageData: StageData, 
  time: number, 
  color: string 
}) {
  const meshRef = useRef<THREE.Mesh>(null);
  
  // Find index for current time
  const idx = useMemo(() => {
    // Simple binary search or find (assume sorted)
    let i = 0;
    while (i < stageData.times.length && stageData.times[i] < time) {
      i++;
    }
    return Math.max(0, i - 1);
  }, [stageData.times, time]);

  // Interpolate position
  const pos = useMemo(() => {
    if (idx >= stageData.times.length - 1) {
      const p = stageData.positions[stageData.positions.length - 1];
      return new THREE.Vector3(p[0]/1000, p[1]/1000, p[2]/1000);
    }
    
    const t0 = stageData.times[idx];
    const t1 = stageData.times[idx+1];
    const alpha = (time - t0) / (t1 - t0);
    
    const p0 = stageData.positions[idx];
    const p1 = stageData.positions[idx+1];
    
    const x = p0[0] + (p1[0] - p0[0]) * alpha;
    const y = p0[1] + (p1[1] - p0[1]) * alpha;
    const z = p0[2] + (p1[2] - p0[2]) * alpha;
    
    return new THREE.Vector3(x/1000, y/1000, z/1000);
  }, [stageData, idx, time]);

  // Calculate velocity for orientation (simplified)
  const vel = useMemo(() => {
    if (idx >= stageData.velocities.length) return new THREE.Vector3(0, 0, 1);
    const v = stageData.velocities[idx];
    return new THREE.Vector3(v[0], v[1], v[2]);
  }, [stageData, idx]);

  useFrame(({ }) => {
    if (meshRef.current) {
      meshRef.current.position.copy(pos);
      
      // Orient rocket along velocity vector if moving fast enough
      if (vel.length() > 10) {
        const target = pos.clone().add(vel.clone().normalize());
        meshRef.current.lookAt(target);
        meshRef.current.rotateX(Math.PI / 2); // Cylinder default is Y-up
      } else {
        // Point radial out
        meshRef.current.lookAt(new THREE.Vector3(0,0,0));
        meshRef.current.rotateX(-Math.PI / 2);
      }
    }
  });

  // Telemetry Data
  const altitude = pos.length() - 6378.137;
  const speed = vel.length();
  const phase = stageData.phases[idx] || 0;

  return (
    <group>
      <mesh ref={meshRef} position={pos}>
        <cylinderGeometry args={[0.2, 0.2, 2, 8]} />
        <meshStandardMaterial color={color} />
      </mesh>
      <Html position={pos} distanceFactor={800}>
        <div style={{ 
          background: 'rgba(0,0,0,0.8)', 
          padding: '4px', 
          borderRadius: '4px',
          border: `1px solid ${color}`,
          color: 'white',
          fontSize: '10px',
          width: '120px',
          fontFamily: 'monospace'
        }}>
          <div><b>{stageData.name}</b></div>
          <div>Alt: {altitude.toFixed(1)} km</div>
          <div>Vel: {speed.toFixed(0)} m/s</div>
          <div>Phase: {phase}</div>
        </div>
      </Html>
    </group>
  );
}

// --- Main App ---

function App() {
  const [data, setData] = useState<FlightData | null>(null);
  const [time, setTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [maxTime, setMaxTime] = useState(100);
  const [playbackSpeed, setPlaybackSpeed] = useState(5);

  useEffect(() => {
    // Fix: Load from the correct relative path for Vite development/production
    // Vite serves 'public' at root '/'
    fetch('flight_data.json')
      .then(res => {
        if (!res.ok) throw new Error("File not found");
        return res.json();
      })
      .then((d: FlightData) => {
        setData(d);
        // Find max time
        let max = 0;
        if (d.stages) {
            d.stages.forEach(s => {
            if (s.times && s.times.length > 0) {
                const last = s.times[s.times.length - 1];
                if (last > max) max = last;
            }
            });
        }
        setMaxTime(max);
      })
      .catch(err => console.error("Failed to load flight data", err));
  }, []);

  // Animation loop
  useEffect(() => {
    let animationFrame: number;
    let lastNow = performance.now();

    const loop = (now: number) => {
      if (isPlaying) {
        const dt = (now - lastNow) / 1000;
        setTime(t => {
          const next = t + dt * playbackSpeed;
          return next > maxTime ? 0 : next;
        });
      }
      lastNow = now;
      animationFrame = requestAnimationFrame(loop);
    };

    animationFrame = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animationFrame);
  }, [isPlaying, maxTime, playbackSpeed]);

  if (!data) return <div style={{color: 'white', padding: 20}}>Loading flight data...</div>;

  // Prepare trajectories
  const trajectories = data.stages.map(stage => {
    return stage.positions.map(p => new THREE.Vector3(p[0]/1000, p[1]/1000, p[2]/1000));
  });

  return (
    <>
      <Canvas camera={{ position: [8000, 0, 0], fov: 45, far: 50000 }}>
        <color attach="background" args={['#050510']} />
        <ambientLight intensity={0.2} />
        <pointLight position={[10000, 0, 0]} intensity={1.5} />
        <Stars radius={20000} depth={50} count={5000} factor={4} fade speed={0} />
        
        <OrbitControls 
          enablePan={true} 
          enableZoom={true} 
          enableRotate={true}
          minDistance={100}
          maxDistance={20000}
        />

        <Earth />
        <LaunchPad position={data.metadata.landing_site} />

        {data.stages.map((stage, i) => (
          <React.Fragment key={stage.id}>
            <TrajectoryLine 
              points={trajectories[i]} 
              color={stage.id.includes('stage1') ? '#ff8800' : '#00ffff'} 
            />
            {/* Only show stage if within its time range */}
            {time >= stage.times[0] && time <= stage.times[stage.times.length-1] && (
              <Rocket 
                stageData={stage} 
                time={time} 
                color={stage.id.includes('stage1') ? '#ff8800' : '#00ffff'}
              />
            )}
          </React.Fragment>
        ))}
      </Canvas>

      {/* UI Overlay */}
      <div className="overlay">
        <h2>🚀 Flight Viz</h2>
        <div>Time: T+{time.toFixed(1)}s</div>
        <div>Speed: {playbackSpeed}x</div>
      </div>

      <div className="controls">
        <button onClick={() => setIsPlaying(!isPlaying)}>
          {isPlaying ? '⏸ Pause' : '▶ Play'}
        </button>
        <input 
          type="range" 
          min={0} 
          max={maxTime} 
          value={time} 
          onChange={(e) => {
            setTime(parseFloat(e.target.value));
            setIsPlaying(false);
          }}
          step={0.1}
        />
        <select 
          value={playbackSpeed} 
          onChange={(e) => setPlaybackSpeed(parseInt(e.target.value))}
          style={{background: '#333', color: 'white', border: '1px solid #555', padding: '8px'}}
        >
          <option value="1">1x</option>
          <option value="5">5x</option>
          <option value="10">10x</option>
          <option value="50">50x</option>
        </select>
      </div>
    </>
  );
}

export default App;
