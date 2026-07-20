import { render, screen } from "@testing-library/react";
import React from "react";
import { describe, expect, it } from "vitest";
import { DemoDataBanner, TechnicalValue } from "@/src/components/ui";

describe("scientific presentation policy", () => {
  it("renders the non-dismissible demo-data warning", () => {
    render(<DemoDataBanner />);
    expect(screen.getByTestId("demo-data-banner")).toHaveTextContent(
      "بيانات تجريبية للعرض — ليست نتائج علمية نهائية",
    );
    expect(screen.queryByRole("button")).not.toBeInTheDocument();
  });

  it("isolates technical hashes as LTR text", () => {
    render(<TechnicalValue value={"02dc0a88decf20367e1a2df6f55d90aab9585d4ac93c1f184f4bd41b472796a7"} copy={false} />);
    expect(screen.getByText(/02dc0a88/)).toHaveAttribute("dir", "ltr");
  });
});
