import { useEffect, useRef, useState } from "preact/hooks";
import type { RoomState } from "./needle-demo-protocol";

interface CozyRoom3DProps {
  readonly state: RoomState;
}

export default function CozyRoom3D({ state }: CozyRoom3DProps) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const stateRef = useRef(state);
  const [ready, setReady] = useState(false);
  stateRef.current = state;

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    let disposed = false;
    let frame = 0;
    let cleanup = () => {};

    void import("three").then((THREE) => {
      if (disposed) return;
      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0xf7dff0);
      scene.fog = new THREE.Fog(0xf7dff0, 14, 28);

      const camera = new THREE.OrthographicCamera(-6, 6, 5, -5, 0.1, 100);
      camera.position.set(10.8, 8.4, 13.8);
      camera.lookAt(0, 2.15, 0);

      const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      renderer.shadowMap.enabled = true;
      renderer.shadowMap.type = THREE.PCFSoftShadowMap;
      renderer.outputColorSpace = THREE.SRGBColorSpace;
      renderer.toneMapping = THREE.ACESFilmicToneMapping;
      renderer.toneMappingExposure = 1.05;
      renderer.domElement.setAttribute("aria-label", "Interactive 3D smart bedroom");
      host.append(renderer.domElement);

      const mat = (color: number, roughness = 0.78) =>
        new THREE.MeshStandardMaterial({ color, roughness, metalness: 0.02 });
      const box = (
        size: [number, number, number],
        color: number,
        position: [number, number, number],
        radius = 0,
      ) => {
        const geometry = radius
          ? new THREE.BoxGeometry(size[0], size[1], size[2], 3, 3, 3)
          : new THREE.BoxGeometry(...size);
        if (radius) {
          const attribute = geometry.attributes.position;
          for (let index = 0; index < attribute.count; index++) {
            const x = attribute.getX(index);
            const y = attribute.getY(index);
            const z = attribute.getZ(index);
            attribute.setXYZ(
              index,
              x - Math.sign(x) * radius * 0.08,
              y - Math.sign(y) * radius * 0.08,
              z - Math.sign(z) * radius * 0.08,
            );
          }
          geometry.computeVertexNormals();
        }
        const mesh = new THREE.Mesh(geometry, mat(color));
        mesh.position.set(...position);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        scene.add(mesh);
        return mesh;
      };

      // Room shell.
      // Match the slab to the walls' outer footprint: extend only beneath the
      // back and left wall thickness, without protruding at the open edges.
      box([9.31, 0.25, 7.41], 0xe7c5b5, [0.345, -0.12, -0.405]);
      const backWall = box([9.2, 6, 0.22], 0xe1c5ca, [0.4, 3, -4]);
      const sideWall = box([0.22, 6, 7.3], 0xd7b9c5, [-4.2, 3, -0.35]);
      for (const wall of [backWall, sideWall]) {
        wall.material.roughness = 1;
        wall.material.metalness = 0;
        wall.material.envMapIntensity = 0;
      }
      const ceiling = new THREE.Mesh(
        new THREE.PlaneGeometry(9.2, 7.3),
        new THREE.MeshStandardMaterial({
          color: 0xffedf5,
          transparent: true,
          opacity: 0.16,
          roughness: 1,
          metalness: 0,
          depthWrite: false,
          side: THREE.DoubleSide,
        }),
      );
      ceiling.position.set(0.4, 6.04, -0.35);
      ceiling.rotation.x = Math.PI / 2;
      scene.add(ceiling);

      const frontWall = new THREE.Mesh(
        new THREE.PlaneGeometry(9.2, 6),
        new THREE.MeshStandardMaterial({
          color: 0xf2c4b8,
          transparent: true,
          opacity: 0.08,
          roughness: 1,
          metalness: 0,
          depthWrite: false,
          side: THREE.DoubleSide,
        }),
      );
      frontWall.position.set(0.4, 3, 3.3);
      scene.add(frontWall);

      const rightWall = new THREE.Mesh(
        new THREE.PlaneGeometry(7.3, 6),
        new THREE.MeshStandardMaterial({
          color: 0xb8d9e7,
          transparent: true,
          opacity: 0.09,
          roughness: 1,
          metalness: 0,
          depthWrite: false,
          side: THREE.DoubleSide,
        }),
      );
      rightWall.position.set(5, 3, -0.35);
      rightWall.rotation.y = Math.PI / 2;
      scene.add(rightWall);

      const door = new THREE.Mesh(
        new THREE.BoxGeometry(0.06, 2.5, 1.5),
        new THREE.MeshStandardMaterial({
          color: 0x9fdcc8,
          transparent: true,
          opacity: 0.28,
          roughness: 0.85,
          metalness: 0,
          depthWrite: false,
        }),
      );
      door.position.set(5, 1.25, 1.65);
      scene.add(door);
      for (const [size, position] of [
        [
          [0.08, 2.65, 0.08],
          [5, 1.325, 0.86],
        ],
        [
          [0.08, 2.65, 0.08],
          [5, 1.325, 2.44],
        ],
        [
          [0.08, 0.08, 1.66],
          [5, 2.61, 1.65],
        ],
      ] as const) {
        const framePart = new THREE.Mesh(
          new THREE.BoxGeometry(...size),
          new THREE.MeshStandardMaterial({
            color: 0xb68aa8,
            transparent: true,
            opacity: 0.42,
            roughness: 0.9,
            metalness: 0,
            depthWrite: false,
          }),
        );
        framePart.position.set(position[0], position[1], position[2]);
        scene.add(framePart);
      }
      const doorKnob = new THREE.Mesh(
        new THREE.SphereGeometry(0.08, 16, 10),
        new THREE.MeshStandardMaterial({
          color: 0xe6b85f,
          transparent: true,
          opacity: 0.55,
          roughness: 0.65,
          metalness: 0,
          depthWrite: false,
        }),
      );
      doorKnob.position.set(5.06, 1.22, 2.15);
      scene.add(doorKnob);

      // Baseboards follow both visible walls and meet cleanly in the corner.
      box([9.2, 0.3, 0.12], 0xffeee8, [0.4, 0.15, -3.84], 0.035);
      box([0.12, 0.3, 7.3], 0xf8e3eb, [-4.04, 0.15, -0.35], 0.035);
      box([9.2, 0.07, 0.17], 0xffffff, [0.4, 0.32, -3.8], 0.025);
      box([0.17, 0.07, 7.3], 0xffffff, [-4, 0.32, -0.35], 0.025);
      box([4.8, 0.06, 4.1], 0xd5a8bd, [0.8, 0.04, -1.84]);

      // Window and animated curtains.
      box([4.2, 2.7, 0.12], 0xafd8e8, [0.8, 3.25, -3.82]);
      box([4.5, 0.16, 0.18], 0xffffff, [0.8, 4.65, -3.68]);
      box([0.12, 2.8, 0.18], 0xffffff, [0.8, 3.25, -3.67]);
      box([4.3, 0.12, 0.18], 0xffffff, [0.8, 1.9, -3.67]);
      const leftCurtain = box([2.15, 3.1, 0.22], 0xcda6e8, [-0.275, 3.25, -3.5], 0.16);
      const rightCurtain = box([2.15, 3.1, 0.22], 0xcda6e8, [1.875, 3.25, -3.5], 0.16);

      // Bed, pillows, and nightstand.
      // Keep the platform in contact with the rug; the previous vertical stack
      // started well above it and made the entire bed appear to float.
      box([4.15, 0.55, 3.8], 0xf4faf9, [0.8, 0.35, -1.99], 0.2);
      box([4.3, 1.05, 0.35], 0xb58d72, [0.8, 0.55, -3.715], 0.12);
      box([3.95, 0.18, 2], 0xf2a9bd, [0.8, 0.68, -1.2], 0.18);
      box([1.2, 0.28, 0.75], 0xfff4dd, [-0.23, 0.75, -3.17], 0.2).rotation.z = -0.08;
      box([1.2, 0.28, 0.75], 0xffe5ef, [1.83, 0.75, -3.17], 0.2).rotation.z = 0.08;
      box([1.05, 0.9, 1.05], 0xa7755d, [4.1, 0.48, -3.365], 0.1);
      box([0.7, 0.08, 0.7], 0xffd277, [4.1, 0.98, -3.365]);

      // Standing fan, beside and aimed toward the center of the bed.
      const fanX = -3.15;
      const fanZ = -2.1;
      box([0.22, 2.1, 0.22], 0x8eb8c7, [fanX, 1.05, fanZ]);
      box([1.25, 0.16, 0.9], 0x709baa, [fanX, 0.12, fanZ], 0.12);
      const fanHead = new THREE.Group();
      fanHead.position.set(fanX, 2.18, fanZ);
      fanHead.rotation.y = Math.atan2(0.8 - fanX, -2.39 - fanZ);
      const fanBlades = new THREE.Group();
      fanBlades.position.z = 0.18;
      for (let index = 0; index < 4; index++) {
        const blade = new THREE.Mesh(new THREE.CapsuleGeometry(0.17, 0.65, 5, 12), mat(0x96cbd1));
        const angle = (Math.PI / 2) * index + 0.22;
        blade.position.set(-Math.sin(angle) * 0.48, Math.cos(angle) * 0.48, 0);
        blade.rotation.z = angle;
        blade.castShadow = true;
        fanBlades.add(blade);
      }
      const fanHub = new THREE.Mesh(new THREE.SphereGeometry(0.2, 24, 16), mat(0xf7c56d));
      fanHub.castShadow = true;
      fanBlades.add(fanHub);
      fanHead.add(fanBlades);
      const fanRing = new THREE.Mesh(new THREE.TorusGeometry(0.92, 0.055, 12, 48), mat(0x5f8996));
      fanRing.position.z = 0.18;
      fanHead.add(fanRing);
      scene.add(fanHead);

      // Ceiling light and controllable illumination.
      const shade = new THREE.Mesh(
        new THREE.ConeGeometry(0.68, 0.55, 32, 1, true),
        new THREE.MeshStandardMaterial({
          color: 0xffd985,
          emissive: 0xffc861,
          emissiveIntensity: 0.4,
          roughness: 0.55,
        }),
      );
      shade.position.set(0.4, 5.05, -0.35);
      shade.castShadow = true;
      scene.add(shade);
      box([0.08, 0.85, 0.08], 0x8a7182, [0.4, 5.62, -0.35]);
      const ceilingCanopy = new THREE.Mesh(
        new THREE.CylinderGeometry(0.24, 0.24, 0.08, 24),
        mat(0x9b7f91),
      );
      ceilingCanopy.position.set(0.4, 5.98, -0.35);
      ceilingCanopy.castShadow = true;
      scene.add(ceilingCanopy);
      const bulb = new THREE.Mesh(
        new THREE.SphereGeometry(0.23, 20, 14),
        new THREE.MeshStandardMaterial({
          color: 0xffe7a5,
          emissive: 0xffc861,
          emissiveIntensity: 1,
        }),
      );
      bulb.position.set(0.4, 4.78, -0.35);
      scene.add(bulb);
      const roomLight = new THREE.PointLight(0xffd69a, 3.2, 14, 1.5);
      roomLight.position.set(0.4, 4.65, -0.35);
      roomLight.castShadow = true;
      roomLight.shadow.mapSize.set(512, 512);
      scene.add(roomLight);

      // Decorative plant, books, wall art, and plush toy.
      box([0.75, 0.65, 0.75], 0xe38f78, [-3.55, 0.35, 2.65], 0.12);
      const leaves = new THREE.Group();
      leaves.position.set(-3.55, 0.82, 2.65);
      for (let index = 0; index < 6; index++) {
        const leaf = new THREE.Mesh(new THREE.SphereGeometry(0.28, 14, 10), mat(0x75aa79));
        const angle = (index / 6) * Math.PI * 2;
        leaf.position.set(Math.cos(angle) * 0.3, 0.2 + (index % 2) * 0.25, Math.sin(angle) * 0.3);
        leaf.scale.set(0.7, 1.4, 0.7);
        leaf.castShadow = true;
        leaves.add(leaf);
      }
      scene.add(leaves);
      box([1.5, 1.15, 0.08], 0xffffff, [-3.25, 3.75, -3.72]);
      box([1.22, 0.87, 0.1], 0xf3b2c8, [-3.25, 3.75, -3.64]);
      box([0.25, 0.38, 0.75], 0x7ba4cf, [3.98, 1.28, -3.365]);
      box([0.25, 0.48, 0.75], 0xe98991, [4.25, 1.33, -3.365]);
      const plush = new THREE.Mesh(new THREE.SphereGeometry(0.42, 20, 14), mat(0xf6bf91));
      plush.position.set(2.3, 0.97, -1);
      plush.castShadow = true;
      scene.add(plush);
      for (const x of [-0.18, 0.18]) {
        const ear = new THREE.Mesh(new THREE.SphereGeometry(0.15, 14, 10), mat(0xf6bf91));
        ear.position.set(2.3 + x, 1.32, -1);
        scene.add(ear);
      }

      const ambient = new THREE.HemisphereLight(0xfff6ed, 0x8e789c, 2.2);
      scene.add(ambient);
      const sun = new THREE.DirectionalLight(0xfff4df, 2.4);
      sun.position.set(5, 9, 7);
      sun.castShadow = true;
      sun.shadow.camera.left = -8;
      sun.shadow.camera.right = 8;
      sun.shadow.camera.top = 8;
      sun.shadow.camera.bottom = -8;
      scene.add(sun);

      let dragging = false;
      let dragStartX = 0;
      let dragStartY = 0;
      let dragStartYaw = 0;
      let dragStartPitch = 0;
      let targetYaw = 0;
      let targetPitch = 0;
      let currentYaw = 0;
      let currentPitch = 0;
      const cameraTarget = new THREE.Vector3(0, 2.15, 0);
      const baseCameraOffset = new THREE.Vector3(10.8, 6.25, 13.8);
      const onPointerDown = (event: PointerEvent) => {
        dragging = true;
        dragStartX = event.clientX;
        dragStartY = event.clientY;
        dragStartYaw = targetYaw;
        dragStartPitch = targetPitch;
        try {
          host.setPointerCapture(event.pointerId);
        } catch {
          // Synthetic pointer events do not have an active pointer to capture.
        }
        renderer.domElement.style.cursor = "grabbing";
      };
      const onPointerMove = (event: PointerEvent) => {
        if (!dragging) return;
        const bounds = host.getBoundingClientRect();
        const deltaX = (event.clientX - dragStartX) / Math.max(1, bounds.width);
        const deltaY = (event.clientY - dragStartY) / Math.max(1, bounds.height);
        targetYaw = THREE.MathUtils.clamp(dragStartYaw - deltaX * 0.42, -0.16, 0.16);
        targetPitch = THREE.MathUtils.clamp(dragStartPitch + deltaY * 0.3, -0.09, 0.09);
      };
      const onPointerRelease = (event: PointerEvent) => {
        if (!dragging) return;
        dragging = false;
        targetYaw = 0;
        targetPitch = 0;
        if (host.hasPointerCapture(event.pointerId)) host.releasePointerCapture(event.pointerId);
        renderer.domElement.style.cursor = "grab";
      };
      host.addEventListener("pointerdown", onPointerDown);
      host.addEventListener("pointermove", onPointerMove);
      host.addEventListener("pointerup", onPointerRelease);
      host.addEventListener("pointercancel", onPointerRelease);

      const resize = () => {
        const width = Math.max(1, host.clientWidth);
        const height = Math.max(1, host.clientHeight);
        const aspect = width / height;
        // Fit the projected room width first. The visual panel is often tall and
        // narrow, so deriving width from a fixed vertical span clipped both walls.
        const halfWidth = 6.9;
        const halfHeight = Math.max(4.75, halfWidth / aspect);
        const horizontalOffset = 0.45;
        const verticalOffset = 1.7;
        camera.left = -halfWidth + horizontalOffset;
        camera.right = halfWidth + horizontalOffset;
        // Bias the view right and upward in world space so the room renders
        // left of center and below the fixed live badge.
        camera.top = halfHeight + verticalOffset;
        camera.bottom = -halfHeight + verticalOffset;
        camera.updateProjectionMatrix();
        renderer.setSize(width, height, false);
      };
      const observer = new ResizeObserver(resize);
      observer.observe(host);
      resize();

      const clock = new THREE.Clock();
      const animate = () => {
        if (disposed) return;
        const delta = Math.min(clock.getDelta(), 0.05);
        const current = stateRef.current;
        const lightTarget = current.lightOn ? 3.4 : 0.15;
        roomLight.intensity += (lightTarget - roomLight.intensity) * 0.08;
        shade.material.emissiveIntensity +=
          ((current.lightOn ? 1.8 : 0.08) - shade.material.emissiveIntensity) * 0.08;
        const leftTarget = current.curtainOpen ? -1.05 : -0.275;
        const rightTarget = current.curtainOpen ? 2.65 : 1.875;
        const curtainScale = current.curtainOpen ? 0.38 : 1;
        leftCurtain.position.x += (leftTarget - leftCurtain.position.x) * 0.08;
        rightCurtain.position.x += (rightTarget - rightCurtain.position.x) * 0.08;
        leftCurtain.scale.x += (curtainScale - leftCurtain.scale.x) * 0.08;
        rightCurtain.scale.x += (curtainScale - rightCurtain.scale.x) * 0.08;
        if (current.fanOn) {
          const speed = current.fanSpeed === "high" ? 10 : current.fanSpeed === "medium" ? 6 : 3;
          fanBlades.rotation.z -= delta * speed;
        }
        const orbitEase = dragging ? 0.16 : 0.035;
        currentYaw += (targetYaw - currentYaw) * orbitEase;
        currentPitch += (targetPitch - currentPitch) * orbitEase;
        const yawCos = Math.cos(currentYaw);
        const yawSin = Math.sin(currentYaw);
        camera.position.set(
          cameraTarget.x + baseCameraOffset.x * yawCos + baseCameraOffset.z * yawSin,
          cameraTarget.y + baseCameraOffset.y + currentPitch * 8,
          cameraTarget.z + baseCameraOffset.z * yawCos - baseCameraOffset.x * yawSin,
        );
        camera.lookAt(cameraTarget);
        renderer.render(scene, camera);
        frame = requestAnimationFrame(animate);
      };
      setReady(true);
      animate();

      cleanup = () => {
        observer.disconnect();
        host.removeEventListener("pointerdown", onPointerDown);
        host.removeEventListener("pointermove", onPointerMove);
        host.removeEventListener("pointerup", onPointerRelease);
        host.removeEventListener("pointercancel", onPointerRelease);
        cancelAnimationFrame(frame);
        scene.traverse((object) => {
          if (object instanceof THREE.Mesh) {
            object.geometry.dispose();
            const materials = Array.isArray(object.material) ? object.material : [object.material];
            for (const material of materials) material.dispose();
          }
        });
        renderer.dispose();
        renderer.domElement.remove();
      };
    });

    return () => {
      disposed = true;
      cleanup();
    };
  }, []);

  return (
    <div
      ref={hostRef}
      class="cozy-room-canvas absolute inset-x-0 top-0 bottom-[4.1rem] min-h-0 touch-pan-y select-none [&_canvas]:block [&_canvas]:h-full [&_canvas]:w-full [&_canvas]:cursor-grab"
    >
      {!ready && (
        <div class="cozy-room-loading absolute inset-0 z-[3] grid place-items-center bg-[#f7dff0] font-bold text-[#765c7c]">
          Building your cozy room…
        </div>
      )}
      <div class="cozy-room-label absolute top-4 left-4 z-[2] flex select-none items-center gap-[0.45rem] rounded-full border-2 border-[rgb(83_62_91_/_0.18)] bg-[rgb(255_250_246_/_0.8)] px-[0.7rem] py-[0.42rem] text-[0.72rem] font-extrabold shadow-[0_0.3rem_0.8rem_rgb(72_46_75_/_0.12)] backdrop-blur-[10px]">
        <span class="cozy-room-live-dot inline-block size-2 rounded-full bg-[#76b88a] shadow-[0_0_0_0.2rem_rgb(118_184_138_/_0.18)]" />{" "}
        Cozy room · live
      </div>
    </div>
  );
}
