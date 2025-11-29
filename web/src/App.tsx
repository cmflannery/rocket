import React, { useState, useEffect, useMemo, useRef, Suspense } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Stars, Sphere, Line, Html } from '@react-three/drei';
import * as THREE from 'three';

// --- Types ---
interface FlightMetadata {
  target_altitude: number;
  landing_site: number[];
  staging_time: number;
  s1_dry_mass: number;
  s1_wet_mass: number;
  s1_landing_propellant: number;
  s2_dry_mass: number;
  s2_wet_mass: number;
  s1_thrust: number;
  s2_thrust: number;
  s1_isp_sl: number;
  s1_isp_vac: number;
  s2_isp: number;
  of_ratio: number;
}

interface FlightData {
  metadata: FlightMetadata;
  stages: StageData[];
}

interface StageData {
  id: string;
  name: string;
  times: number[];
  positions: number[][];
  velocities: number[][];
  phases: number[];
  masses?: number[];
  thrusts?: number[];
  accelerations?: number[];
}

// Phase names for display
const PHASE_NAMES: Record<number, string> = {
  1: 'ASCENT',
  2: 'COAST',
  3: 'BOOSTBACK',
  4: 'ENTRY',
  5: 'DESCENT',
  6: 'LANDING',
  7: 'LANDED',
  10: 'COAST',
  11: 'BURN',
  12: 'ORBIT',
};

// --- Styles ---
const FONT = '"IBM Plex Mono", monospace';
const COLORS = {
  bg: '#0a0a0a',
  panel: 'rgba(10, 10, 10, 0.9)',
  border: '#333',
  accent: '#00cc99',
  text: '#e0e0e0',
  textDim: '#666',
  stage1: '#ff6600',
  stage2: '#0099ff',
};

// --- Components ---

// Coordinate axes at origin (ECI frame) - labels are screen-space
function CoordinateAxes({ size = 8000 }: { size?: number }) {
  const labelStyle = {
    fontWeight: 600, 
    fontSize: '11px', 
    fontFamily: FONT,
    textTransform: 'uppercase' as const,
    letterSpacing: '0.05em',
    textShadow: '0 0 4px rgba(0,0,0,0.8)',
  };
  
  return (
    <group>
      {/* X axis - Red */}
      <Line points={[[0,0,0], [size,0,0]]} color="#cc0000" lineWidth={2} />
      <Html position={[size * 1.02, 0, 0]} center>
        <div style={{ ...labelStyle, color: '#cc0000' }}>X</div>
      </Html>
      
      {/* Y axis - Green */}
      <Line points={[[0,0,0], [0,size,0]]} color="#00cc00" lineWidth={2} />
      <Html position={[0, size * 1.02, 0]} center>
        <div style={{ ...labelStyle, color: '#00cc00' }}>Y</div>
      </Html>
      
      {/* Z axis - Blue - points to North Pole */}
      <Line points={[[0,0,0], [0,0,size]]} color="#0066cc" lineWidth={2} />
      <Html position={[0, 0, size * 1.02]} center>
        <div style={{ ...labelStyle, color: '#0099ff' }}>Z [NORTH]</div>
      </Html>
    </group>
  );
}

// Earth's rotation rate in rad/s
const OMEGA_EARTH = 7.2921159e-5;

function GroundTrack({ stageData, color }: { stageData: StageData, color: string }) {
  const points = useMemo(() => {
    return stageData.positions.map((p, i) => {
      const t = stageData.times[i];
      // Earth rotation angle at time t
      const theta = OMEGA_EARTH * t;
      
      // ECI coordinates (km)
      const x_eci = p[0] / 1000;
      const y_eci = p[1] / 1000;
      const z_eci = p[2] / 1000;
      
      // Transform to ECEF (rotate by -theta around Z)
      // P_ecef = Rot(-theta) * P_eci
      const c = Math.cos(-theta);
      const s = Math.sin(-theta);
      
      const x_ecef = x_eci * c - y_eci * s;
      const y_ecef = x_eci * s + y_eci * c;
      const z_ecef = z_eci;
      
      // Project to surface (plus small offset to avoid z-fighting)
      const r = Math.sqrt(x_ecef*x_ecef + y_ecef*y_ecef + z_ecef*z_ecef);
      const R_EARTH = 6378.137;
      const offset = 10.0; // 10km above surface
      const scale = (R_EARTH + offset) / r;
      
      return new THREE.Vector3(x_ecef * scale, y_ecef * scale, z_ecef * scale);
    });
  }, [stageData]);

  return (
    <Line 
      points={points} 
      color={color} 
      lineWidth={1.5} 
      opacity={0.6} 
      transparent 
      dashed={false}
    />
  );
}

function EarthWithLaunchPad({ time, launchSite, children }: { time: number, launchSite: number[], children?: React.ReactNode }) {
  const colorMap = useMemo(() => {
    const loader = new THREE.TextureLoader();
    loader.crossOrigin = 'anonymous';
    return loader.load('https://unpkg.com/three-globe@2.31.0/example/img/earth-blue-marble.jpg');
  }, []);
  
  // Earth rotates counter-clockwise around Z axis when viewed from above North Pole
  const earthRotation = OMEGA_EARTH * time;
  
  // Launch pad position in ECEF (fixed to Earth's surface)
  // This is the position at t=0, which stays fixed relative to Earth
  const padPos = useMemo(() => new THREE.Vector3(
    launchSite[0] / 1000,
    launchSite[1] / 1000,
    launchSite[2] / 1000
  ), [launchSite]);

  // The entire group rotates around Z axis
  // The Earth texture needs an additional X rotation to align poles with Z
  return (
    <group rotation={[0, 0, earthRotation]}>
      {/* Earth sphere - tilted so poles align with Z axis */}
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
      
      {/* Launch pad - fixed to Earth's surface */}
      <ScreenSpaceMarker position={padPos} color="#00cc66" size={4} />
      <Html position={padPos} center style={{ pointerEvents: 'none' }}>
        <div style={{ 
          color: '#00cc66', 
          fontWeight: 600, 
          fontSize: '10px', 
          fontFamily: FONT,
          textTransform: 'uppercase',
          letterSpacing: '0.05em',
          textShadow: '0 0 4px rgba(0,0,0,0.9)',
          whiteSpace: 'nowrap',
          marginTop: '-20px',
        }}>
          LAUNCH SITE
        </div>
      </Html>

      {/* Ground Tracks and other Earth-fixed children */}
      {children}
    </group>
  );
}

function ClickableOrbitControls({ onTargetChange }: { onTargetChange: (target: THREE.Vector3) => void }) {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const controlsRef = useRef<any>(null);
  const { camera, gl, scene } = useThree();
  const raycaster = useMemo(() => new THREE.Raycaster(), []);
  
  useEffect(() => {
    const handleClick = (event: MouseEvent) => {
      if (event.detail !== 2) return;
      
      const rect = gl.domElement.getBoundingClientRect();
      const mouse = new THREE.Vector2(
        ((event.clientX - rect.left) / rect.width) * 2 - 1,
        -((event.clientY - rect.top) / rect.height) * 2 + 1
      );
      
      raycaster.setFromCamera(mouse, camera);
      const intersects = raycaster.intersectObjects(scene.children, true);
      
      if (intersects.length > 0) {
        const point = intersects[0].point;
        if (controlsRef.current) {
          controlsRef.current.target.copy(point);
          controlsRef.current.update();
          onTargetChange(point);
        }
      }
    };
    
    gl.domElement.addEventListener('click', handleClick);
    return () => gl.domElement.removeEventListener('click', handleClick);
  }, [camera, gl, scene, raycaster, onTargetChange]);

  useFrame(() => {
    if (controlsRef.current) {
      camera.up.set(0, 0, 1);
    }
  });

  return (
    <OrbitControls 
      ref={controlsRef}
      enablePan={true} 
      enableZoom={true} 
      enableRotate={true}
      enableDamping={false}
      minDistance={100}
      maxDistance={50000}
      zoomToCursor={true}
      zoomSpeed={0.5}
    />
  );
}

// Screen-space sized marker component
function ScreenSpaceMarker({ position, color, size = 8 }: { position: THREE.Vector3, color: string, size?: number }) {
  const meshRef = useRef<THREE.Mesh>(null);
  const { camera } = useThree();
  
  useFrame(() => {
    if (meshRef.current) {
      // Scale based on distance to camera to maintain constant screen size
      const distance = camera.position.distanceTo(position);
      const scale = distance * 0.003 * size; // Adjust multiplier for desired screen size
      meshRef.current.scale.setScalar(scale);
    }
  });
  
  return (
    <mesh ref={meshRef} position={position}>
      <sphereGeometry args={[1, 12, 12]} />
      <meshBasicMaterial color={color} />
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
  const lineRef = useRef<any>(null); // Ref for the radial line
  const { camera } = useThree();
  
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

  // Get thrust status
  const thrust = stageData.thrusts?.[idx] || 0;
  const isBurning = thrust > 1000; // More than 1kN = burning
  
  // Color changes when thrusting: bright yellow/white when burning
  const displayColor = isBurning ? '#ffff00' : color;

  // Scale marker based on camera distance for constant screen size
  useFrame(() => {
    if (meshRef.current) {
      meshRef.current.position.copy(pos);
      
      // Scale based on distance to camera - smaller markers
      const distance = camera.position.distanceTo(pos);
      const baseSize = isBurning ? 6 : 4; // Slightly bigger when burning
      const scale = distance * 0.003 * baseSize;
      meshRef.current.scale.setScalar(scale);
      
      if (vel.length() > 10) {
        const target = pos.clone().add(vel.clone().normalize());
        meshRef.current.lookAt(target);
        meshRef.current.rotateX(Math.PI / 2);
      } else {
        meshRef.current.lookAt(new THREE.Vector3(0,0,0));
        meshRef.current.rotateX(-Math.PI / 2);
      }
      
      // Update material color dynamically
      const material = meshRef.current.material as THREE.MeshBasicMaterial;
      material.color.set(displayColor);
    }
  });

  const altitude = pos.length() - 6378.137;
  const speed = vel.length();
  const phase = stageData.phases[idx] || 0;
  const phaseName = PHASE_NAMES[phase] || `PHASE ${phase}`;

  return (
    <group>
      {/* Rocket mesh - screen-space sized marker */}
      <mesh ref={meshRef} position={pos}>
        <sphereGeometry args={[1, 12, 12]} />
        <meshBasicMaterial color={displayColor} />
      </mesh>
      
      {/* Radial Line (Dotted) */}
      <Line
        points={[pos, new THREE.Vector3(0, 0, 0)]}
        color={color}
        lineWidth={1}
        dashed={true}
        dashScale={10}  // Adjust dash size
        gapSize={5}     // Adjust gap size
        opacity={0.4}
        transparent
      />

      {/* Screen-space telemetry label */}
      <Html position={pos} style={{ pointerEvents: 'none' }}>
        <div style={{ 
          background: COLORS.panel, 
          padding: '6px 8px', 
          border: `1px solid ${isBurning ? '#ffff00' : color}`,
          color: COLORS.text,
          fontSize: '10px',
          fontFamily: FONT,
          lineHeight: 1.4,
          marginLeft: '15px',
          marginTop: '-35px',
          whiteSpace: 'nowrap',
        }}>
          <div style={{ 
            fontWeight: 600, 
            textTransform: 'uppercase', 
            marginBottom: '2px', 
            color: isBurning ? '#ffff00' : color 
          }}>
            {stageData.name.toUpperCase()} {isBurning ? '[BURN]' : ''}
          </div>
          <div>ALT: {altitude.toFixed(1)} KM</div>
          <div>VEL: {speed.toFixed(0)} M/S</div>
          <div>PHASE: {phaseName}</div>
        </div>
      </Html>
    </group>
  );
}

// --- Telemetry Components ---

function PropellantBar({ 
  label, 
  current, 
  max, 
  color 
}: { 
  label: string, 
  current: number, 
  max: number, 
  color: string 
}) {
  const pct = Math.max(0, Math.min(100, (current / max) * 100));
  const isLow = pct < 20;
  
  return (
    <div style={{ marginBottom: '6px' }}>
      <div style={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        fontSize: '10px',
        marginBottom: '2px',
      }}>
        <span style={{ color: COLORS.textDim }}>{label}</span>
        <span style={{ color: isLow ? '#ff4444' : COLORS.text }}>{pct.toFixed(1)}%</span>
      </div>
      <div style={{ 
        height: '6px', 
        background: '#1a1a1a', 
        border: `1px solid ${COLORS.border}`,
      }}>
        <div style={{ 
          height: '100%', 
          width: `${pct}%`, 
          background: isLow ? '#ff4444' : color,
          transition: 'width 0.1s',
        }} />
      </div>
    </div>
  );
}

function StagePanel({ 
  stage, 
  time, 
  meta, 
  isStage1,
  landingSite 
}: { 
  stage: StageData, 
  time: number, 
  meta: FlightMetadata,
  isStage1: boolean,
  landingSite: number[]
}) {
  const color = isStage1 ? COLORS.stage1 : COLORS.stage2;
  const stageData = getStageDataAtTime(stage, time);
  const isActive = time >= stage.times[0] && time <= stage.times[stage.times.length - 1];
  
  // Simple propellant calculation: mass - dry_mass = propellant remaining
  // Data now correctly records each stage's mass independently
  const dryMass = isStage1 ? meta.s1_dry_mass : meta.s2_dry_mass;
  const wetMass = isStage1 ? meta.s1_wet_mass : meta.s2_wet_mass;
  const maxPropellant = wetMass - dryMass;
  const currentPropellant = Math.max(0, stageData.mass - dryMass);
  
  // O/F ratio for LOX/RP-1 is typically 2.56
  const ofRatio = meta.of_ratio || 2.56;
  const oxFraction = ofRatio / (1 + ofRatio);
  const fuelFraction = 1 / (1 + ofRatio);
  
  const currentOx = currentPropellant * oxFraction;
  const currentFuel = currentPropellant * fuelFraction;
  const maxOx = maxPropellant * oxFraction;
  const maxFuel = maxPropellant * fuelFraction;
  
  // Calculate G-load
  const gLoad = stageData.accel / 9.80665;
  
  // Max thrust reference
  const maxThrust = isStage1 ? meta.s1_thrust : meta.s2_thrust;
  const throttle = stageData.thrust > 0 ? (stageData.thrust / maxThrust) * 100 : 0;
  
  const phaseName = PHASE_NAMES[stageData.phase] || `PHASE ${stageData.phase}`;
  const isBurning = stageData.thrust > 0;
  
  // Calculate distance to landing site (for S1 RTLS)
  // Landing site rotates with Earth, so we need to account for that
  const padTheta = OMEGA_EARTH * time;
  const padX = landingSite[0] * Math.cos(padTheta) - landingSite[1] * Math.sin(padTheta);
  const padY = landingSite[0] * Math.sin(padTheta) + landingSite[1] * Math.cos(padTheta);
  const padZ = landingSite[2];
  
  const dx = stageData.position[0] - padX;
  const dy = stageData.position[1] - padY;
  const dz = stageData.position[2] - padZ;
  const distanceToPad = Math.sqrt(dx*dx + dy*dy + dz*dz) / 1000; // km
  
  // Also calculate horizontal distance (ground track)
  const stageR = Math.sqrt(stageData.position[0]**2 + stageData.position[1]**2 + stageData.position[2]**2);
  const padR = Math.sqrt(padX**2 + padY**2 + padZ**2);
  // Project both to surface and compute arc distance
  const dotProduct = (stageData.position[0]*padX + stageData.position[1]*padY + stageData.position[2]*padZ) / (stageR * padR);
  const angleRad = Math.acos(Math.max(-1, Math.min(1, dotProduct)));
  const groundDistance = angleRad * 6378.137; // km along surface
  
  return (
    <div style={{ 
      marginBottom: '16px',
      opacity: isActive ? 1 : 0.4,
    }}>
      {/* Stage Header */}
      <div style={{ 
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        borderBottom: `1px solid ${color}`,
        paddingBottom: '6px',
        marginBottom: '8px',
      }}>
        <span style={{ 
          fontWeight: 600, 
          color,
          textTransform: 'uppercase',
          letterSpacing: '0.05em',
        }}>
          {stage.name}
        </span>
        <span style={{ 
          fontSize: '10px',
          padding: '2px 6px',
          background: isBurning ? color : 'transparent',
          color: isBurning ? '#000' : COLORS.textDim,
          border: `1px solid ${isBurning ? color : COLORS.border}`,
        }}>
          {phaseName}
        </span>
      </div>
      
      {/* Flight Data */}
      <div style={{ 
        display: 'grid', 
        gridTemplateColumns: '1fr 1fr',
        gap: '4px 12px',
        fontSize: '11px',
        marginBottom: '10px',
      }}>
        <div>
          <span style={{ color: COLORS.textDim }}>ALT:</span> {stageData.altitude.toFixed(1)} KM
        </div>
        <div>
          <span style={{ color: COLORS.textDim }}>VEL:</span> {stageData.speed.toFixed(0)} M/S
        </div>
        <div>
          <span style={{ color: COLORS.textDim }}>MASS:</span> {formatMass(stageData.mass)}
        </div>
        <div>
          <span style={{ color: COLORS.textDim }}>G:</span> {gLoad.toFixed(2)}
        </div>
        {isStage1 && (
          <>
            <div style={{ gridColumn: '1 / -1', marginTop: '4px', borderTop: `1px solid ${COLORS.border}`, paddingTop: '4px' }}>
              <span style={{ color: COLORS.textDim }}>DIST TO PAD:</span> {distanceToPad.toFixed(1)} KM
            </div>
            <div>
              <span style={{ color: COLORS.textDim }}>GROUND:</span> {groundDistance.toFixed(1)} KM
            </div>
          </>
        )}
      </div>
      
      {/* Thrust */}
      <div style={{ marginBottom: '10px' }}>
        <div style={{ 
          display: 'flex', 
          justifyContent: 'space-between',
          fontSize: '10px',
          marginBottom: '2px',
        }}>
          <span style={{ color: COLORS.textDim }}>THRUST</span>
          <span>{formatThrust(stageData.thrust)} ({throttle.toFixed(0)}%)</span>
        </div>
        <div style={{ 
          height: '6px', 
          background: '#1a1a1a', 
          border: `1px solid ${COLORS.border}`,
        }}>
          <div style={{ 
            height: '100%', 
            width: `${throttle}%`, 
            background: isBurning ? color : COLORS.border,
            transition: 'width 0.1s',
          }} />
        </div>
      </div>
      
      {/* Propellant Tanks */}
      <div style={{ fontSize: '10px', marginBottom: '4px', color: COLORS.textDim }}>
        PROPELLANT
      </div>
      <PropellantBar 
        label="LOX" 
        current={currentOx} 
        max={maxOx} 
        color="#66ccff" 
      />
      <PropellantBar 
        label="RP-1" 
        current={currentFuel} 
        max={maxFuel} 
        color="#ffaa44" 
      />
      
      {/* Propellant mass remaining */}
      <div style={{ 
        fontSize: '10px', 
        color: COLORS.textDim,
        marginTop: '4px',
      }}>
        REMAINING: {formatMass(currentPropellant)}
      </div>
    </div>
  );
}

function StageTelemetryPanel({ data, time }: { data: FlightData, time: number }) {
  const stage1 = data.stages.find(s => s.id === 'stage1');
  const stage2 = data.stages.find(s => s.id === 'stage2');
  
  return (
    <div style={{
      position: 'absolute',
      top: 16,
      right: 16,
      background: COLORS.panel,
      padding: '16px 20px',
      border: `1px solid ${COLORS.border}`,
      color: COLORS.text,
      fontFamily: FONT,
      fontSize: '12px',
      width: '260px',
      maxHeight: 'calc(100vh - 100px)',
      overflowY: 'auto',
    }}>
      <div style={{ 
        fontSize: '14px', 
        fontWeight: 600, 
        marginBottom: '16px',
        textTransform: 'uppercase',
        letterSpacing: '0.1em',
        borderBottom: `1px solid ${COLORS.border}`,
        paddingBottom: '8px',
      }}>
        STAGE TELEMETRY
      </div>
      
      {stage1 && (
        <StagePanel 
          stage={stage1} 
          time={time} 
          meta={data.metadata}
          isStage1={true}
          landingSite={data.metadata.landing_site}
        />
      )}
      
      {stage2 && (
        <StagePanel 
          stage={stage2} 
          time={time} 
          meta={data.metadata}
          isStage1={false}
          landingSite={data.metadata.landing_site}
        />
      )}
      
      {/* Vehicle Summary */}
      <div style={{ 
        borderTop: `1px solid ${COLORS.border}`,
        paddingTop: '12px',
        marginTop: '8px',
      }}>
        <div style={{ 
          fontSize: '10px', 
          fontWeight: 600, 
          marginBottom: '8px',
          textTransform: 'uppercase',
          color: COLORS.textDim,
        }}>
          VEHICLE CONFIG
        </div>
        <div style={{ 
          display: 'grid', 
          gridTemplateColumns: '1fr 1fr',
          gap: '4px',
          fontSize: '10px',
          color: COLORS.textDim,
        }}>
          <div>S1 DRY: {formatMass(data.metadata.s1_dry_mass || 0)}</div>
          <div>S2 DRY: {formatMass(data.metadata.s2_dry_mass || 0)}</div>
          <div>S1 ISP: {data.metadata.s1_isp_vac || 0}S</div>
          <div>S2 ISP: {data.metadata.s2_isp || 0}S</div>
        </div>
      </div>
    </div>
  );
}

// --- Helper functions ---

function getStageDataAtTime(stage: StageData, time: number) {
  let idx = 0;
  while (idx < stage.times.length && stage.times[idx] < time) {
    idx++;
  }
  idx = Math.max(0, idx - 1);
  
  const pos = stage.positions[idx] || [0, 0, 0];
  const vel = stage.velocities[idx] || [0, 0, 0];
  const altitude = (Math.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2) / 1000) - 6378.137;
  const speed = Math.sqrt(vel[0]**2 + vel[1]**2 + vel[2]**2);
  const phase = stage.phases?.[idx] || 0;
  const mass = stage.masses?.[idx] || 0;
  const thrust = stage.thrusts?.[idx] || 0;
  const accel = stage.accelerations?.[idx] || 0;
  
  return { idx, altitude, speed, phase, mass, thrust, accel, position: pos };
}

function formatMass(kg: number): string {
  if (kg >= 1000) return `${(kg / 1000).toFixed(1)} T`;
  return `${kg.toFixed(0)} KG`;
}

function formatThrust(n: number): string {
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)} MN`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(0)} KN`;
  return `${n.toFixed(0)} N`;
}

// --- Main App ---

function App() {
  const [data, setData] = useState<FlightData | null>(null);
  const [time, setTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [maxTime, setMaxTime] = useState(100);
  const [playbackSpeed, setPlaybackSpeed] = useState(5);
  const [pivotPoint, setPivotPoint] = useState<THREE.Vector3 | null>(null);

  useEffect(() => {
    fetch('flight_data.json')
      .then(res => {
        if (!res.ok) throw new Error("File not found");
        return res.json();
      })
      .then((d: FlightData) => {
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

  if (!data) return (
    <div style={{
      color: COLORS.text, 
      padding: 40, 
      background: COLORS.bg, 
      height: '100vh',
      fontFamily: FONT,
      fontSize: '14px',
      textTransform: 'uppercase',
      letterSpacing: '0.1em'
    }}>
      LOADING FLIGHT DATA...
    </div>
  );

  const trajectories = data.stages.map(stage => {
    return stage.positions.map(p => new THREE.Vector3(p[0]/1000, p[1]/1000, p[2]/1000));
  });

  return (
    <div style={{ width: '100vw', height: '100vh', position: 'relative', background: COLORS.bg }}>
      <Canvas 
        camera={{ position: [15000, 0, 0], fov: 45, far: 100000, up: [0, 0, 1] }}
        style={{ width: '100%', height: '100%' }}
      >
        <color attach="background" args={['#050508']} />
        <ambientLight intensity={1.5} />
        <directionalLight position={[15000, 10000, 10000]} intensity={2} />
        <directionalLight position={[-15000, -5000, 5000]} intensity={0.8} />
        <Stars radius={20000} depth={50} count={5000} factor={4} fade speed={0} />
        
        <ClickableOrbitControls onTargetChange={setPivotPoint} />
        <CoordinateAxes size={8000} />

        <Suspense fallback={null}>
          <EarthWithLaunchPad time={time} launchSite={data.metadata.landing_site}>
            {data.stages.map(stage => (
              <GroundTrack 
                key={stage.id} 
                stageData={stage} 
                color={stage.id.includes('stage1') ? COLORS.stage1 : COLORS.stage2} 
              />
            ))}
          </EarthWithLaunchPad>
        </Suspense>

        {data.stages.map((stage, i) => (
          <React.Fragment key={stage.id}>
            <TrajectoryLine 
              points={trajectories[i]} 
              color={stage.id.includes('stage1') ? COLORS.stage1 : COLORS.stage2} 
            />
            {time >= stage.times[0] && time <= stage.times[stage.times.length-1] && (
              <Rocket 
                stageData={stage} 
                time={time} 
                color={stage.id.includes('stage1') ? COLORS.stage1 : COLORS.stage2}
              />
            )}
          </React.Fragment>
        ))}
      </Canvas>

      {/* LEFT PANEL - Mission Timer */}
      <div style={{
        position: 'absolute',
        top: 16,
        left: 16,
        background: COLORS.panel,
        padding: '16px 20px',
        border: `1px solid ${COLORS.border}`,
        color: COLORS.text,
        fontFamily: FONT,
        fontSize: '12px',
        minWidth: '180px',
      }}>
        <div style={{ 
          fontSize: '14px', 
          fontWeight: 600, 
          marginBottom: '12px',
          textTransform: 'uppercase',
          letterSpacing: '0.1em',
          borderBottom: `1px solid ${COLORS.border}`,
          paddingBottom: '8px',
        }}>
          MISSION CONTROL
        </div>
        <div style={{ marginBottom: '4px' }}>
          <span style={{ color: COLORS.textDim }}>MET:</span> T+{time.toFixed(1)}S
        </div>
        <div style={{ marginBottom: '4px' }}>
          <span style={{ color: COLORS.textDim }}>RATE:</span> {playbackSpeed}X
        </div>
        <div style={{ marginBottom: '4px' }}>
          <span style={{ color: COLORS.textDim }}>STAGING:</span> T+{data.metadata.staging_time?.toFixed(0) || '?'}S
        </div>
        {pivotPoint && (
          <div style={{ marginTop: '12px', fontSize: '10px', color: COLORS.textDim }}>
            PIVOT: [{pivotPoint.x.toFixed(0)}, {pivotPoint.y.toFixed(0)}, {pivotPoint.z.toFixed(0)}]
          </div>
        )}
        <div style={{ marginTop: '8px', fontSize: '10px', color: COLORS.textDim }}>
          DBL-CLICK: SET PIVOT
        </div>
      </div>

      {/* RIGHT PANEL - Stage Telemetry */}
      <StageTelemetryPanel data={data} time={time} />

      {/* PLAYBACK CONTROLS */}
      <div style={{
        position: 'absolute',
        bottom: 20,
        left: '50%',
        transform: 'translateX(-50%)',
        background: COLORS.panel,
        padding: '12px 16px',
        display: 'flex',
        gap: '12px',
        alignItems: 'center',
        border: `1px solid ${COLORS.border}`,
      }}>
        <button 
          onClick={() => setIsPlaying(!isPlaying)}
          style={{
            background: isPlaying ? '#660000' : '#006600',
            color: COLORS.text,
            border: `1px solid ${isPlaying ? '#990000' : '#009900'}`,
            padding: '8px 16px',
            cursor: 'pointer',
            fontFamily: FONT,
            fontWeight: 600,
            fontSize: '12px',
            textTransform: 'uppercase',
            letterSpacing: '0.05em',
          }}
        >
          {isPlaying ? 'STOP' : 'PLAY'}
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
          style={{ 
            width: '280px', 
            cursor: 'pointer',
            accentColor: COLORS.accent,
          }}
        />
        <select 
          value={playbackSpeed} 
          onChange={(e) => setPlaybackSpeed(parseInt(e.target.value))}
          style={{
            background: '#1a1a1a',
            color: COLORS.text,
            border: `1px solid ${COLORS.border}`,
            padding: '8px 12px',
            fontFamily: FONT,
            fontSize: '12px',
            cursor: 'pointer',
            textTransform: 'uppercase',
          }}
        >
          <option value="1">1X</option>
          <option value="5">5X</option>
          <option value="10">10X</option>
          <option value="50">50X</option>
        </select>
      </div>
    </div>
  );
}

export default App;
