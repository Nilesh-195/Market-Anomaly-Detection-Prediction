import React, { useRef, useMemo } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Environment, Float, Instance, Instances } from '@react-three/drei'
import * as THREE from 'three'

function BoxSculpture({ count = 100 }) {
  const ref = useRef()
  
  // Generate a grid of points
  const points = useMemo(() => {
    const arr = []
    const size = Math.sqrt(count)
    for (let i = 0; i < size; i++) {
      for (let j = 0; j < size; j++) {
        arr.push({
          x: (i - size / 2) * 0.4 + (Math.random() * 0.1),
          z: (j - size / 2) * 0.4 + (Math.random() * 0.1),
          delay: Math.random() * Math.PI * 2,
          scale: 0.5 + Math.random() * 2,
          // Mix of navy/blue/cyan colors for market aesthetic
          color: new THREE.Color().setHSL(0.55 + Math.random() * 0.1, 0.8, 0.2 + Math.random() * 0.5)
        })
      }
    }
    return arr
  }, [count])

  useFrame((state) => {
    const time = state.clock.getElapsedTime()
    if (ref.current) {
      ref.current.children.forEach((child, i) => {
        const point = points[i]
        // Animate Y scale and position to simulate market waves
        const wave = Math.sin(point.x * 2 + time + point.delay) * Math.cos(point.z * 2 + time * 0.5)
        const currentScale = point.scale + wave * 0.5
        const targetScaleY = Math.max(0.1, currentScale)
        
        child.scale.set(0.15, targetScaleY, 0.15)
        child.position.set(point.x, targetScaleY / 2, point.z)
        
        // Enhance highlight color if it spikes
        if (wave > 0.8) {
          child.color.setHSL(0.5, 1, 0.8) // Cyan spike
        } else {
          child.color.copy(point.color)
        }
      })
    }
    // Gentle overall rotation
    if (ref.current) {
      ref.current.rotation.y = Math.sin(time * 0.1) * 0.1
    }
  })

  return (
    <Instances ref={ref} range={points.length} material={new THREE.MeshStandardMaterial({ roughness: 0.2, metalness: 0.8 })}>
      <boxGeometry args={[1, 1, 1]} />
      {points.map((props, i) => (
        <Instance key={i} color={props.color} />
      ))}
    </Instances>
  )
}

export default function MarketPulseSculpture() {
  return (
    <div className="w-full h-full min-h-[400px] relative bg-gradient-to-b from-blue-50/20 to-transparent rounded-2xl overflow-hidden [&>canvas]:outline-none">
      <Canvas camera={{ position: [4, 3, 5], fov: 45 }} dpr={[1, 2]}>
        <ambientLight intensity={0.5} />
        <directionalLight position={[10, 10, 5]} intensity={1} castShadow />
        <directionalLight position={[-10, 5, -5]} intensity={0.5} color="#00ffff" />
        
        <Float speed={2} rotationIntensity={0.2} floatIntensity={0.5}>
          <BoxSculpture count={144} />
        </Float>

        <Environment preset="city" />
        <OrbitControls 
          enableZoom={false} 
          enablePan={false} 
          minPolarAngle={Math.PI / 4} 
          maxPolarAngle={Math.PI / 2.5}
          autoRotate
          autoRotateSpeed={0.5}
        />
      </Canvas>
      
      {/* Small legend overlay */}
      <div className="absolute bottom-4 left-4 flex gap-3 text-[10px] font-mono text-brand-blue/70 bg-white/80 backdrop-blur px-3 py-1.5 rounded-full border border-card-border shadow-sm">
        <div className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-cyan-400 shadow-[0_0_8px_rgba(34,211,238,0.8)]"></span>
          <span>Anomaly detected</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-[#1e3a8a]"></span>
          <span>Market baseline</span>
        </div>
      </div>
    </div>
  )
}
