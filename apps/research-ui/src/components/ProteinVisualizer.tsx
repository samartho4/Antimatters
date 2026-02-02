import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import { ProteinNode } from '../types';

interface VisualizerProps {
  dataString?: string;
  isAnnotationMode: boolean;
  onAnnotationComplete: (selectedNodes: any[], snapshotUrl: string) => void;
}

export const ProteinVisualizer: React.FC<VisualizerProps> = ({ 
  dataString, 
  isAnnotationMode,
  onAnnotationComplete 
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const wrapperRef = useRef<HTMLDivElement>(null);
  const [nodes, setNodes] = useState<ProteinNode[]>([]);
  
  // Initial / Updates
  useEffect(() => {
    let parsedNodes: ProteinNode[] = [];
    if (dataString) {
        try {
            // Try to extract JSON from block
            const jsonMatch = dataString.match(/```json:simulation([\s\S]*?)```/);
            const jsonStr = jsonMatch ? jsonMatch[1] : dataString;
            const data = JSON.parse(jsonStr);
            parsedNodes = data.nodes || [];
        } catch (e) {
            // Fallback or incremental
        }
    }

    if (parsedNodes.length === 0) {
        // Generate advanced demo data (IDP Cloud)
        parsedNodes = Array.from({ length: 40 }, (_, i) => ({
            id: i,
            x: 400 + (Math.random() - 0.5) * 150,
            y: 300 + (Math.random() - 0.5) * 150,
            type: i > 35 ? 'pocket' : 'backbone',
            residue: ['ALA', 'GLY', 'SER', 'PRO'][Math.floor(Math.random() * 4)],
            radius: i > 35 ? 15 : 6, // Pockets are larger invisible zones
            vx: 0, vy: 0
        }));
        // Add Ligand
        parsedNodes.push({ id: 999, x: 200, y: 200, type: 'ligand', residue: 'LIG', radius: 10 });
    }
    setNodes(parsedNodes);
  }, [dataString]);

  // Simulation & Rendering
  useEffect(() => {
    if (!svgRef.current || nodes.length === 0) return;

    const width = wrapperRef.current?.clientWidth || 800;
    const height = wrapperRef.current?.clientHeight || 600;

    // Simulation for IDP "Cloud" dynamics
    const simulation = d3.forceSimulation(nodes)
      .force("charge", d3.forceManyBody().strength((d: any) => d.type === 'pocket' ? 0 : -20))
      .force("center", d3.forceCenter(width / 2, height / 2).strength(0.05))
      .force("collide", d3.forceCollide().radius((d: any) => (d.radius || 5) + 2))
      .force("x", d3.forceX(width/2).strength(0.01))
      .force("y", d3.forceY(height/2).strength(0.01));

    // Links (Backbone connectivity)
    const links = [];
    for (let i = 0; i < nodes.length - 1; i++) {
        if (nodes[i].type === 'backbone' && nodes[i+1].type === 'backbone') {
            links.push({ source: nodes[i], target: nodes[i+1] });
        }
    }
    simulation.force("link", d3.forceLink(links).distance(15).strength(0.8));

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove(); // Clear

    // 1. Render "Cloud" Hull (The Ensemble representation)
    const hullGroup = svg.append("g").attr("class", "hull-layer");
    
    // 2. Render Links
    const linkGroup = svg.append("g").attr("class", "links");
    
    // 3. Render Nodes
    const nodeGroup = svg.append("g").attr("class", "nodes");
    
    // 4. Annotation Layer (Brush)
    const brushGroup = svg.append("g").attr("class", "brush");

    // Elements
    const link = linkGroup.selectAll("line").data(links).join("line")
      .attr("stroke", "#e2e8f0").attr("stroke-width", 1.5).attr("opacity", 0.6);

    const node = nodeGroup.selectAll("circle").data(nodes).join("circle")
      .attr("r", (d: any) => d.radius || 5)
      .attr("fill", (d: any) => {
          if (d.type === 'ligand') return '#f43f5e'; // Red
          if (d.type === 'pocket') return 'url(#pocketGradient)'; // Cryptic Pocket
          return '#3b82f6'; // Protein Blue
      })
      .attr("fill-opacity", (d: any) => d.type === 'pocket' ? 0.4 : 1)
      .attr("stroke", "#fff").attr("stroke-width", 1);

    // Gradients
    const defs = svg.append("defs");
    const grad = defs.append("radialGradient").attr("id", "pocketGradient");
    grad.append("stop").attr("offset", "0%").attr("stop-color", "#fbbf24").attr("stop-opacity", 0.6); // Amber center
    grad.append("stop").attr("offset", "100%").attr("stop-color", "#fbbf24").attr("stop-opacity", 0);

    // Simulation Tick
    simulation.on("tick", () => {
      // Constraints
      nodes.forEach((d: any) => {
         d.x = Math.max(10, Math.min(width - 10, d.x));
         d.y = Math.max(10, Math.min(height - 10, d.y));
      });

      link
        .attr("x1", (d: any) => d.source.x)
        .attr("y1", (d: any) => d.source.y)
        .attr("x2", (d: any) => d.target.x)
        .attr("y2", (d: any) => d.target.y);

      node
        .attr("cx", (d: any) => d.x)
        .attr("cy", (d: any) => d.y);

      // Draw Hull (Ensemble Cloud)
      const backboneNodes = nodes.filter((n: any) => n.type === 'backbone').map((n: any) => [n.x, n.y]);
      if (backboneNodes.length > 3) {
          const hull = d3.polygonHull(backboneNodes as [number, number][]);
          if (hull) {
              const hullPath = "M" + hull.join("L") + "Z";
              hullGroup.selectAll("path")
                  .data([hullPath])
                  .join("path")
                  .attr("d", d => d)
                  .attr("fill", "#bae6fd") // Light blue cloud
                  .attr("fill-opacity", 0.2)
                  .attr("stroke", "none")
                  .transition().duration(100).ease(d3.easeLinear); // Smooth update
          }
      }
    });

    // Brush Logic (Annotation)
    if (isAnnotationMode) {
        const brush = d3.brush()
            .extent([[0, 0], [width, height]])
            .on("end", (event) => {
                if (!event.selection) return;
                const [[x0, y0], [x1, y1]] = event.selection;
                
                // Find selected nodes
                const selected = nodes.filter((d: any) => 
                    d.x >= x0 && d.x <= x1 && d.y >= y0 && d.y <= y1
                );
                
                if (selected.length > 0) {
                    // Create a visual snapshot (simulated here with a color indicator, 
                    // in prod we'd use foreignObject or canvas.toDataURL)
                    const snapshotUrl = "data:image/svg+xml;base64,..."; // Placeholder for valid base64
                    
                    // Highlight selected temporarily
                    node.filter((d:any) => selected.includes(d))
                        .attr("stroke", "#f59e0b")
                        .attr("stroke-width", 3);

                    onAnnotationComplete(selected, snapshotUrl);
                    
                    // Clear brush after short delay
                    setTimeout(() => {
                        brushGroup.call(brush.move as any, null);
                    }, 500);
                }
            });

        brushGroup.call(brush as any);
    } else {
        brushGroup.on(".brush", null);
        brushGroup.selectAll("*").remove();
    }

    return () => { simulation.stop(); };
  }, [nodes, isAnnotationMode]);

  return (
    <div ref={wrapperRef} className="w-full h-full bg-slate-900 relative overflow-hidden rounded-xl border border-slate-700 shadow-2xl">
        <div className="absolute top-4 left-4 z-10 pointer-events-none">
             <div className="flex flex-col gap-2">
                <div className="bg-black/50 backdrop-blur p-2 rounded border border-white/10 text-[10px] font-mono text-cyan-400">
                    <p>● ENSEMBLE CLOUD: ACTIVE</p>
                    <p>● CRYPTIC POCKETS: DETECTED</p>
                    <p>● TEMP: 310K | PH: 7.4</p>
                </div>
                {isAnnotationMode && (
                    <div className="bg-amber-500/20 backdrop-blur p-2 rounded border border-amber-500/50 text-[10px] font-mono text-amber-200 animate-pulse">
                        ⚠ ANNOTATION MODE: SELECT REGION
                    </div>
                )}
             </div>
        </div>
      <svg ref={svgRef} className={`w-full h-full ${isAnnotationMode ? 'cursor-crosshair' : 'cursor-default'}`} />
    </div>
  );
};