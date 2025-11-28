import React, { useState, useEffect, useMemo, useRef, Suspense } from 'react';
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
  const colorMap = useMemo(() => {
    const loader = new THREE.TextureLoader();
    loader.crossOrigin = 'anonymous';
    return loader.load('https://unpkg.com/three-globe@2.31.0/example/img/earth-blue-marble.jpg');
  }, []);

  // Rotate poles from Y-axis to Z-axis to match ECI frame
  return (
    <Sphere 
      args={[6378.137, 64, 64]} 
      position={[0, 0, 0]}
      rotation={[Math.PI / 2, 0, 0]}
    >
      <meshStandardMaterial 
        map={colorMap}
        roughness={0.5}
        metalness={0.0}
      />
    </Sphere>
  );
}

function LaunchPad({ position }: { position: number[] }) {
  const pos = new THREE.Vector3(position[0]/1000, position[1]/1000, position[2]/1000);
  
  // Create a visible marker that sticks out from the surface
  const markerHeight = 100; // km above surface for visibility
  const direction = pos.clone().normalize();
  const markerPos = pos.clone().add(direction.multiplyScalar(markerHeight));
  
  return (
    <group>
      {/* Marker at surface */}
      <mesh position={pos}>
        <sphereGeometry args={[30, 16, 16]} />
        <meshBasicMaterial color="#00ff00" />
      </mesh>
      {/* Tall beacon for visibility */}
      <mesh position={markerPos}>
        <coneGeometry args={[20, 80, 8]} />
        <meshBasicMaterial color="#ffff00" />
      </mesh>
      {/* Line connecting them */}
      <Line 
        points={[pos, markerPos]}
        color="#00ff00"
        lineWidth={2}
      />
    </group>
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
  
  const idx = useMemo(() => {
    let i = 0;
    while (i < stageData.times.length && stageData.times[i] < time) {
      i++;
    }
    return Math.max(0, i - 1);
  }, [stageData.times, time]);

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

  const vel = useMemo(() => {
    if (idx >= stageData.velocities.length) return new THREE.Vector3(0, 0, 1);
    const v = stageData.velocities[idx];
    return new THREE.Vector3(v[0], v[1], v[2]);
  }, [stageData, idx]);

  useFrame(() => {
    if (meshRef.current) {
      meshRef.current.position.copy(pos);
      
      if (vel.length() > 10) {
        const target = pos.clone().add(vel.clone().normalize());
        meshRef.current.lookAt(target);
        meshRef.current.rotateX(Math.PI / 2);
      } else {
        meshRef.current.lookAt(new THREE.Vector3(0,0,0));
        meshRef.current.rotateX(-Math.PI / 2);
      }
    }
  });

  const altitude = pos.length() - 6378.137;
  const speed = vel.length();
  const phase = stageData.phases[idx] || 0;

  return (
    <group>
      <mesh ref={meshRef} position={pos}>
        <cylinderGeometry args={[1, 1, 10, 8]} />
        <meshStandardMaterial color={color} />
      </mesh>
      <Html position={pos} distanceFactor={5000}>
        <div style={{ 
          background: 'rgba(0,0,0,0.8)', 
          padding: '4px', 
          borderRadius: '4px',
          border: `1px solid ${color}`,
          color: 'white',
          fontSize: '12px',
          width: '150px',
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
    console.log("Fetching flight_data.json...");
    fetch('flight_data.json')
      .then(res => {
        if (!res.ok) throw new Error("File not found");
        return res.json();
      })
      .then((d: FlightData) => {
        console.log("Data loaded:", d);
        setData(d);
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

  if (!data) return <div style={{color: 'white', padding: 20, background: '#111', height: '100vh'}}>Loading flight data...</div>;

  const trajectories = data.stages.map(stage => {
    return stage.positions.map(p => new THREE.Vector3(p[0]/1000, p[1]/1000, p[2]/1000));
  });

  return (
    <div style={{ width: '100vw', height: '100vh', position: 'relative' }}>
      <Canvas 
        camera={{ position: [15000, 0, 0], fov: 45, far: 100000 }}
        style={{ width: '100%', height: '100%' }}
      >
        <color attach="background" args={['#050510']} />
        <ambientLight intensity={1.5} />
        <directionalLight position={[15000, 10000, 10000]} intensity={2} />
        <directionalLight position={[-15000, -5000, 5000]} intensity={0.8} />
        <Stars radius={20000} depth={50} count={5000} factor={4} fade speed={0} />
        
        <OrbitControls 
          enablePan={true} 
          enableZoom={true} 
          enableRotate={true}
          enableDamping={false}
          minDistance={100}
          maxDistance={50000}
          zoomToCursor={true}
          zoomSpeed={0.5}
        />

        <Suspense fallback={null}>
          <Earth />
        </Suspense>
        
        <LaunchPad position={data.metadata.landing_site} />

        {data.stages.map((stage, i) => (
          <React.Fragment key={stage.id}>
            <TrajectoryLine 
              points={trajectories[i]} 
              color={stage.id.includes('stage1') ? '#ff8800' : '#00ffff'} 
            />
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

      {/* Telemetry Overlay - Top Left */}
      <div style={{
        position: 'absolute',
        top: 20,
        left: 20,
        background: 'rgba(0, 0, 0, 0.7)',
        padding: '15px 20px',
        borderRadius: '8px',
        border: '1px solid #00ffff',
        color: 'white',
        fontFamily: 'monospace',
        fontSize: '14px',
        pointerEvents: 'none'
      }}>
        <div style={{ fontSize: '18px', fontWeight: 'bold', marginBottom: '8px' }}>🚀 Flight Viz</div>
        <div>Time: T+{time.toFixed(1)}s</div>
        <div>Speed: {playbackSpeed}x</div>
      </div>

      {/* Playback Controls - Bottom Center */}
      <div style={{
        position: 'absolute',
        bottom: 30,
        left: '50%',
        transform: 'translateX(-50%)',
        background: 'rgba(0, 0, 0, 0.8)',
        padding: '12px 20px',
        borderRadius: '30px',
        display: 'flex',
        gap: '15px',
        alignItems: 'center',
        border: '1px solid #444'
      }}>
        <button 
          onClick={() => setIsPlaying(!isPlaying)}
          style={{
            background: isPlaying ? '#ff4444' : '#44ff44',
            color: 'black',
            border: 'none',
            padding: '10px 20px',
            cursor: 'pointer',
            borderRadius: '20px',
            fontFamily: 'monospace',
            fontWeight: 'bold',
            fontSize: '14px'
          }}
        >
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
          style={{ width: '300px', cursor: 'pointer' }}
        />
        <select 
          value={playbackSpeed} 
          onChange={(e) => setPlaybackSpeed(parseInt(e.target.value))}
          style={{
            background: '#222',
            color: 'white',
            border: '1px solid #555',
            padding: '8px 12px',
            borderRadius: '8px',
            fontFamily: 'monospace',
            cursor: 'pointer'
          }}
        >
          <option value="1">1x</option>
          <option value="5">5x</option>
          <option value="10">10x</option>
          <option value="50">50x</option>
        </select>
      </div>
    </div>
  );
}

export default App;
