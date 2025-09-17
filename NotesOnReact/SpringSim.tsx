/* This is a demo where you can drag the icon and just have it snap back to center as if it was
bound by a spring. User can update the spring coefficient too.

*/
import { useState, useEffect, useRef } from "react";
import logo from "./assets/react.svg";

export default function App() {
  const center = useRef({
    x: window.innerWidth / 2,
    y: window.innerHeight / 2,
  });

  const [pos, setPos] = useState(center.current);
  const [dragging, setDragging] = useState(false);
  const [returning, setReturning] = useState(false);
  const [k, setK] = useState(0.3); // spring stiffness (slider controlled)

  const velocity = useRef({ x: 0, y: 0 });
  const requestRef = useRef(null);

  // mouse drag handling
  useEffect(() => {
    const handleMouseMove = (e) => {
      if (dragging) {
        setPos({ x: e.clientX, y: e.clientY });
        velocity.current = { x: 0, y: 0 };
      }
    };

    const handleMouseUp = () => {
      setDragging(false);
      setReturning(true);
    };

    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);

    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
    };
  }, [dragging]);

  // spring physics loop
  useEffect(() => {
    if (!returning) return;

    const c = 0.1; // damping

    const animate = () => {
      setPos((prev) => {
        const dx = center.current.x - prev.x;
        const dy = center.current.y - prev.y;

        const ax = k * dx - c * velocity.current.x;
        const ay = k * dy - c * velocity.current.y;

        velocity.current.x += ax;
        velocity.current.y += ay;

        return {
          x: prev.x + velocity.current.x,
          y: prev.y + velocity.current.y,
        };
      });

      requestRef.current = requestAnimationFrame(animate);
    };

    requestRef.current = requestAnimationFrame(animate);

    return () => cancelAnimationFrame(requestRef.current);
  }, [returning, k]);
// JSX to draw up the page
  return (
    <div
      style={{
        height: "100vh",
        backgroundColor: "#f8fafdff",
        position: "relative",
        overflow: "hidden",
      }}
    >
      {/* Header */}
      <div
        style={{
          position: "absolute",
          top: "10px",
          left: "50%",
          transform: "translateX(-50%)",
          fontSize: "24px",
          fontWeight: "bold",
          color: "#333",
          userSelect: "none",
        }}
      >
        Spring Demo
      </div>

      {/* Logo */}
      <img
        src={logo}
        alt="React logo"
        draggable={false}
        onDragStart={(e) => e.preventDefault()}
        style={{
          width: "160px",
          height: "160px",
          position: "absolute",
          left: pos.x - 80,
          top: pos.y - 80,
          cursor: "grab",
          userSelect: "none",
        }}
        onMouseDown={() => {
          setDragging(true);
          setReturning(false);
        }}
      />

      {/* Slider UI */}
      <div
        style={{
          position: "absolute",
          bottom: "20px",
          left: "50%",
          transform: "translateX(-50%)",
          textAlign: "center",
        }}
      >
        <input
          type="range"
          min="0.1"
          max="1"
          step="0.01"
          value={k}
          onChange={(e) => setK(Number(e.target.value))}
        />
        <div>Spring stiffness k: {k.toFixed(2)}</div>
      </div>
    </div>
  );
}
