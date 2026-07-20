# SynthDet UI design system

## Brand and tokens

`BrandLogo` uses the exact supplied bitmap and never redraws the symbol. Semantic CSS variables
cover background, foreground, card, primary, secondary, muted, accent, success, warning,
destructive, border, input, ring, and seven chart colours. Light mode uses the specified royal
blue/navy/cyan/sky palette; dark mode has separately designed navy surfaces rather than inversion.

Surfaces use fine borders, restrained shadows, low-opacity grid motifs, and controlled gradients.
Blur, glass effects, radii, and continuous motion are deliberately limited.

## Typography and bidirectionality

Tajawal is loaded at its published 300, 400, 500, 700, and 800 faces; CSS weight 600 resolves
between 500/700 because Tajawal does not publish a 600 file. Arabic stays RTL. Hashes, paths,
commands, model IDs, and numeric identifiers use `bdi`, `dir="ltr"`, and monospace.

## Components and accessibility

The system includes `AppShell`, `BrandLogo`, `PageHeader`, `MetricCard`, `StatusBadge`,
`TechnicalValue`, `IdentityCard`, `PendingScientificResults`, `DemoDataBanner`, charts,
`ExperimentTable`, composition bars, timelines, inference controls, and analysis galleries.

Transitions remain 150–300 ms, initial content is never hidden, and reduced-motion disables
nonessential animation. Status combines icon and text; controls have labels, keyboard focus, and
semantic roles.
