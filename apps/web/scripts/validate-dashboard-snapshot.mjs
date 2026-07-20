import fs from "node:fs";
import path from "node:path";

const file = path.resolve("src/data/generated/project-snapshot.json");
const snapshot = JSON.parse(fs.readFileSync(file, "utf8"));
const fail = (message) => {
  throw new Error(`Invalid dashboard snapshot: ${message}`);
};

if (snapshot.schemaVersion !== 2) fail("wrong schema version");
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
if (snapshot.scientificResults.available !== true) fail("sealed final results must be available");
if (snapshot.scientificResults.finalTestEvaluated !== true) fail("final test campaign must be complete");
if (snapshot.scientificResults.recommendedModel !== "real_only") fail("unexpected preregistered winner");
if (snapshot.scientificResults.ranking.length !== 5) fail("final ranking must contain five regimes");
if (snapshot.experiments.some((regime) => regime.status !== "completed")) fail("all regimes must be complete");
if (snapshot.training.testSetAccessCount !== 0) fail("test set access must remain zero");
if (snapshot.audit.authorizedEvaluationCampaigns !== 1) fail("evaluation access audit changed");
if (snapshot.audit.resultHashesVerified !== 5 || snapshot.audit.predictionHashesVerified !== 10) {
  fail("sealed hash counts changed");
}

const serialized = JSON.stringify(snapshot);
for (const forbidden of ["image_path", "label_path", "datasets/raw", "models/", ".pt"]) {
  if (serialized.includes(forbidden)) fail(`forbidden content: ${forbidden}`);
}

console.log("Dashboard snapshot validation passed: sealed five-model results, audit, and protected content policy verified.");
