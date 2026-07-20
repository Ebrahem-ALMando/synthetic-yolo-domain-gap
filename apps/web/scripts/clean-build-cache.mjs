import fs from "node:fs";
import path from "node:path";

for (const relative of [".next", "tsconfig.tsbuildinfo"]) {
  const target = path.resolve(relative);
  if (path.dirname(target) !== process.cwd() && target !== path.resolve(".next")) {
    throw new Error(`Refusing to clean outside apps/web: ${target}`);
  }
  fs.rmSync(target, { recursive: true, force: true });
}
