import fs from "node:fs";
import path from "node:path";

const file = path.resolve("src/data/generated/project-snapshot.json");
const snapshot = JSON.parse(fs.readFileSync(file, "utf8"));
const fail = (message) => {
  throw new Error(`Invalid dashboard snapshot: ${message}`);
};

if (snapshot.schemaVersion !== 1) fail("wrong schema version");
if (snapshot.source !== "repository") fail("source must be repository");
if (snapshot.dataset.classCount !== 7) fail("class count must be seven");
if (snapshot.experiments.length !== 5) fail("five regimes are required");
if (snapshot.dataset.splits.train !== 427 || snapshot.dataset.splits.val !== 140) {
  fail("active train/validation counts changed");
}
if (snapshot.dataset.splits.test !== 68 || snapshot.dataset.protectedTest !== true) {
  fail("protected-test metadata contract changed");
}
if (snapshot.audit.protectedContentIncluded !== false) fail("protected content exposed");
if (snapshot.scientificResults.available !== false) fail("final results must remain unavailable");
if (snapshot.training.testSetAccessCount !== 0) fail("test set access must remain zero");

const serialized = JSON.stringify(snapshot);
for (const forbidden of ["image_path", "label_path", "datasets/raw", "models/", ".pt"]) {
  if (serialized.includes(forbidden)) fail(`forbidden content: ${forbidden}`);
}

console.log("Dashboard snapshot validation passed: 7 classes, 5 regimes, protected content absent.");
