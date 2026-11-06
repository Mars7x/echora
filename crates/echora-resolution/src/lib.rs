//! Resolution planning shared by every graphics backend.
//!
//! The game renders for a fixed *guest logical* size. Echora renders the
//! scene at an independently chosen *internal* size, composes it into a
//! *presentation rectangle* of the *output* surface, and keeps the UI in its
//! own coordinate space. These are distinct quantities; nothing here assumes
//! an integer relationship between them.
//!
//! A [`ResolutionPlan`] answers the questions a backend needs while it
//! translates guest work: what size a guest render target becomes on the host
//! ([`ResolutionPlan::map_target`]), where a guest viewport or scissor lands
//! ([`ResolutionPlan::map_rect`]), and which resources a settings change
//! invalidates ([`ResolutionPlan::impact_of_change`]).

use std::fmt;

/// Largest dimension accepted for any surface. Backends may impose a lower,
/// device-reported limit through [`ResolutionRequest::max_dimension`].
pub const MAX_DIMENSION: u32 = 32768;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Extent {
    pub width: u32,
    pub height: u32,
}

impl Extent {
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    pub fn aspect(self) -> Ratio {
        Ratio::new(self.width, self.height)
    }

    fn validate(self, what: &'static str, max: u32) -> Result<Self, ResolutionError> {
        if self.width == 0 || self.height == 0 || self.width > max || self.height > max {
            return Err(ResolutionError::InvalidExtent {
                what,
                extent: self,
                max,
            });
        }
        Ok(self)
    }
}

impl fmt::Display for Extent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}x{}", self.width, self.height)
    }
}

/// Signed rectangle in pixels (guest viewports may start off-screen).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

/// An exact positive rational, kept reduced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Ratio {
    pub num: u32,
    pub den: u32,
}

impl Ratio {
    /// # Panics
    /// If either term is zero.
    pub fn new(num: u32, den: u32) -> Self {
        assert!(num != 0 && den != 0, "ratio terms must be nonzero");
        let g = gcd(num, den);
        Self {
            num: num / g,
            den: den / g,
        }
    }

    pub fn as_f64(self) -> f64 {
        f64::from(self.num) / f64::from(self.den)
    }
}

impl fmt::Display for Ratio {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.num, self.den)
    }
}

fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// `v * num / den` rounded to nearest, ties away from zero, in 128-bit
/// arithmetic so no intermediate overflows.
fn scale_round(v: i64, num: u64, den: u64) -> i64 {
    let p = i128::from(v) * i128::from(num);
    let d = i128::from(den);
    let q = (p.abs() + d / 2) / d;
    (if p < 0 { -q } else { q }) as i64
}

/// How the internal scene size is chosen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InternalResolution {
    /// Render at the presentation rectangle's size: one scene pixel per
    /// output pixel.
    MatchOutput,
    /// Render at this height; width follows the scene aspect ratio. "1440p"
    /// style selection, independent of the window size.
    Height(u32),
    /// Render at this fraction of the presentation rectangle (for example
    /// 3/4 for a 75% render scale).
    Scale(Ratio),
    /// Exact scene size. Its aspect ratio must match the scene's to within
    /// one pixel of rounding.
    Exact(Extent),
}

/// How the scene's aspect ratio relates to the game's and the display's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AspectPolicy {
    /// Keep the guest aspect ratio; letterbox or pillarbox on other displays.
    Original,
    /// Widen (or heighten) the view to fill the output's aspect ratio. This
    /// needs game-specific projection and UI adjustments; the plan reports it
    /// through [`ResolutionPlan::expanded_view`].
    Expand,
    /// A specific aspect ratio, fitted into the output.
    Fixed(Ratio),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolutionRequest {
    /// The size the game believes it renders at.
    pub guest_logical: Extent,
    /// The drawable size of the output surface in physical pixels.
    pub output: Extent,
    pub internal: InternalResolution,
    pub aspect: AspectPolicy,
    /// Device limit on image dimensions; `None` uses [`MAX_DIMENSION`].
    pub max_dimension: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResolutionError {
    InvalidExtent {
        what: &'static str,
        extent: Extent,
        max: u32,
    },
    AspectMismatch {
        requested: Extent,
        scene_aspect: Ratio,
    },
}

impl fmt::Display for ResolutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidExtent { what, extent, max } => {
                write!(
                    f,
                    "{what} size {extent} is invalid (each side must be 1..={max})"
                )
            }
            Self::AspectMismatch {
                requested,
                scene_aspect,
            } => write!(
                f,
                "internal resolution {requested} does not have the scene aspect ratio {scene_aspect}"
            ),
        }
    }
}

impl std::error::Error for ResolutionError {}

/// How a guest render target scales.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetClass {
    /// Sized relative to the screen (color/depth buffers, half- or
    /// quarter-resolution effect buffers): scales with the scene.
    ScreenRelative,
    /// Sized independently of the screen (shadow maps, lookup tables,
    /// textures): keeps its guest size unless a pass policy says otherwise.
    Fixed,
}

/// Which resources a change from one plan to another invalidates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ChangeImpact {
    None,
    /// Only the presentation (swapchain size or placement) changes.
    Presentation,
    /// Screen-relative render targets, history buffers and dependent
    /// constants must be recreated; game state is unaffected.
    SceneTargets,
}

/// A resolved set of sizes. Construct with [`ResolutionPlan::new`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolutionPlan {
    pub guest_logical: Extent,
    /// The guest-space extent the scene covers. Equal to `guest_logical`
    /// unless the aspect policy expands the view.
    pub guest_view: Extent,
    pub scene_aspect: Ratio,
    /// Internal scene render size.
    pub scene: Extent,
    pub output: Extent,
    /// Where the composed scene is placed in the output.
    pub presentation: Rect,
    /// Whether the view is wider or taller than the game's own.
    pub expanded_view: bool,
}

impl ResolutionPlan {
    pub fn new(req: &ResolutionRequest) -> Result<Self, ResolutionError> {
        let max = req
            .max_dimension
            .unwrap_or(MAX_DIMENSION)
            .min(MAX_DIMENSION);
        let guest = req.guest_logical.validate("guest logical", max)?;
        let output = req.output.validate("output", max)?;

        let scene_aspect = match req.aspect {
            AspectPolicy::Original => guest.aspect(),
            AspectPolicy::Expand => output.aspect(),
            AspectPolicy::Fixed(r) => r,
        };
        let presentation = fit(scene_aspect, output);

        // The guest-space region the scene shows. Expansion keeps the guest's
        // height (horizontal widening) or width (vertical) and extends the
        // other axis, so UI and projections can be adjusted by one factor.
        let guest_view = if scene_aspect == guest.aspect() {
            guest
        } else if scene_aspect.as_f64() > guest.aspect().as_f64() {
            Extent::new(
                scale_round(
                    i64::from(guest.height),
                    u64::from(scene_aspect.num),
                    u64::from(scene_aspect.den),
                ) as u32,
                guest.height,
            )
        } else {
            Extent::new(
                guest.width,
                scale_round(
                    i64::from(guest.width),
                    u64::from(scene_aspect.den),
                    u64::from(scene_aspect.num),
                ) as u32,
            )
        };

        let present_extent = Extent::new(presentation.width, presentation.height);
        let scene = match req.internal {
            InternalResolution::MatchOutput => present_extent,
            InternalResolution::Height(h) => Extent::new(
                scale_round(
                    i64::from(h),
                    u64::from(scene_aspect.num),
                    u64::from(scene_aspect.den),
                ) as u32,
                h,
            ),
            InternalResolution::Scale(s) => Extent::new(
                scale_round(
                    i64::from(present_extent.width),
                    u64::from(s.num),
                    u64::from(s.den),
                ) as u32,
                scale_round(
                    i64::from(present_extent.height),
                    u64::from(s.num),
                    u64::from(s.den),
                ) as u32,
            ),
            InternalResolution::Exact(e) => {
                let expect_w = scale_round(
                    i64::from(e.height),
                    u64::from(scene_aspect.num),
                    u64::from(scene_aspect.den),
                );
                if (i64::from(e.width) - expect_w).abs() > 1 {
                    return Err(ResolutionError::AspectMismatch {
                        requested: e,
                        scene_aspect,
                    });
                }
                e
            }
        }
        .validate("internal scene", max)?;

        Ok(Self {
            guest_logical: guest,
            guest_view,
            scene_aspect,
            scene,
            output,
            presentation,
            expanded_view: guest_view != guest,
        })
    }

    /// Host size of a guest render target. Screen-relative targets scale by
    /// the exact scene/guest ratio on each axis and never collapse to zero.
    pub fn map_target(&self, guest: Extent, class: TargetClass) -> Extent {
        match class {
            TargetClass::Fixed => guest,
            TargetClass::ScreenRelative => {
                let (sx, sy) = self.guest_to_scene_scale();
                Extent::new(
                    (scale_round(i64::from(guest.width), sx.0, sx.1) as u32).max(1),
                    (scale_round(i64::from(guest.height), sy.0, sy.1) as u32).max(1),
                )
            }
        }
    }

    /// Maps a guest viewport or scissor rectangle on a screen-relative target
    /// into scene pixels. Both edges are rounded independently, so rectangles
    /// that tile the guest target also tile the host target without gaps or
    /// overlaps.
    pub fn map_rect(&self, r: Rect) -> Rect {
        let (sx, sy) = self.guest_to_scene_scale();
        let x0 = scale_round(i64::from(r.x), sx.0, sx.1);
        let y0 = scale_round(i64::from(r.y), sy.0, sy.1);
        let x1 = scale_round(i64::from(r.x) + i64::from(r.width), sx.0, sx.1);
        let y1 = scale_round(i64::from(r.y) + i64::from(r.height), sy.0, sy.1);
        Rect {
            x: x0 as i32,
            y: y0 as i32,
            width: (x1 - x0) as u32,
            height: (y1 - y0) as u32,
        }
    }

    /// Classifies a guest target by its size: a target whose sides are the
    /// guest view divided by the same small power of two (1, 2, 4, 8, 16) on
    /// both axes is treated as screen-relative. Backends should prefer
    /// explicit, game-verified classifications where they exist.
    pub fn classify(&self, guest: Extent) -> TargetClass {
        let v = self.guest_view;
        let rel = (0..5).any(|shift| {
            let d = 1u32 << shift;
            guest.width == v.width.div_ceil(d) && guest.height == v.height.div_ceil(d)
                || guest.width == v.width / d && guest.height == v.height / d
        });
        if rel {
            TargetClass::ScreenRelative
        } else {
            TargetClass::Fixed
        }
    }

    /// Exact per-axis scene/guest-view scale factors as (num, den).
    pub fn guest_to_scene_scale(&self) -> ((u64, u64), (u64, u64)) {
        (
            (
                u64::from(self.scene.width),
                u64::from(self.guest_view.width),
            ),
            (
                u64::from(self.scene.height),
                u64::from(self.guest_view.height),
            ),
        )
    }

    pub fn impact_of_change(&self, next: &Self) -> ChangeImpact {
        if self.scene != next.scene
            || self.guest_view != next.guest_view
            || self.guest_logical != next.guest_logical
        {
            ChangeImpact::SceneTargets
        } else if self.output != next.output || self.presentation != next.presentation {
            ChangeImpact::Presentation
        } else {
            ChangeImpact::None
        }
    }
}

/// Largest rectangle of `aspect` centred in `outer`.
fn fit(aspect: Ratio, outer: Extent) -> Rect {
    let (a, b) = (u64::from(aspect.num), u64::from(aspect.den));
    let (w, h) = (u64::from(outer.width), u64::from(outer.height));
    let (fw, fh) = if w * b >= h * a {
        // Output is wider: full height, pillarbox.
        (scale_round(h as i64, a, b) as u64, h)
    } else {
        (w, scale_round(w as i64, b, a) as u64)
    };
    let (fw, fh) = (fw.clamp(1, w), fh.clamp(1, h));
    Rect {
        x: ((w - fw) / 2) as i32,
        y: ((h - fh) / 2) as i32,
        width: fw as u32,
        height: fh as u32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GUEST: Extent = Extent::new(1920, 1080);

    fn plan(output: Extent, internal: InternalResolution, aspect: AspectPolicy) -> ResolutionPlan {
        ResolutionPlan::new(&ResolutionRequest {
            guest_logical: GUEST,
            output,
            internal,
            aspect,
            max_dimension: None,
        })
        .unwrap()
    }

    #[test]
    fn non_integer_scale_renders_at_true_size() {
        let p = plan(
            Extent::new(1600, 900),
            InternalResolution::MatchOutput,
            AspectPolicy::Original,
        );
        assert_eq!(p.scene, Extent::new(1600, 900));
        // 5/6 scale: half-resolution guest buffers become 800x450.
        assert_eq!(
            p.map_target(Extent::new(960, 540), TargetClass::ScreenRelative),
            Extent::new(800, 450)
        );
        let p = plan(
            Extent::new(2560, 1440),
            InternalResolution::MatchOutput,
            AspectPolicy::Original,
        );
        assert_eq!(
            p.map_target(GUEST, TargetClass::ScreenRelative),
            Extent::new(2560, 1440)
        );
        assert_eq!(
            p.map_target(Extent::new(480, 270), TargetClass::ScreenRelative),
            Extent::new(640, 360)
        );
    }

    #[test]
    fn internal_resolution_is_independent_of_output() {
        let p = plan(
            Extent::new(3840, 2160),
            InternalResolution::Height(1440),
            AspectPolicy::Original,
        );
        assert_eq!(p.scene, Extent::new(2560, 1440));
        assert_eq!(
            p.presentation,
            Rect {
                x: 0,
                y: 0,
                width: 3840,
                height: 2160
            }
        );
        let p = plan(
            Extent::new(2560, 1440),
            InternalResolution::Scale(Ratio::new(3, 4)),
            AspectPolicy::Original,
        );
        assert_eq!(p.scene, Extent::new(1920, 1080));
        let p = plan(
            Extent::new(1280, 720),
            InternalResolution::Exact(Extent::new(3840, 2160)),
            AspectPolicy::Original,
        );
        assert_eq!(p.scene, Extent::new(3840, 2160));
    }

    #[test]
    fn exact_resolution_must_match_scene_aspect() {
        let err = ResolutionPlan::new(&ResolutionRequest {
            guest_logical: GUEST,
            output: Extent::new(1920, 1080),
            internal: InternalResolution::Exact(Extent::new(1920, 1200)),
            aspect: AspectPolicy::Original,
            max_dimension: None,
        })
        .unwrap_err();
        assert!(matches!(err, ResolutionError::AspectMismatch { .. }));
    }

    #[test]
    fn ultrawide_original_aspect_pillarboxes() {
        let p = plan(
            Extent::new(3440, 1440),
            InternalResolution::MatchOutput,
            AspectPolicy::Original,
        );
        assert_eq!(p.scene, Extent::new(2560, 1440));
        assert_eq!(
            p.presentation,
            Rect {
                x: 440,
                y: 0,
                width: 2560,
                height: 1440
            }
        );
        assert!(!p.expanded_view);
    }

    #[test]
    fn ultrawide_expand_widens_guest_view() {
        let p = plan(
            Extent::new(3440, 1440),
            InternalResolution::MatchOutput,
            AspectPolicy::Expand,
        );
        assert_eq!(p.scene, Extent::new(3440, 1440));
        assert_eq!(p.guest_view, Extent::new(2580, 1080));
        assert!(p.expanded_view);
        // A full-view guest target and its half-size buffer both scale by 4/3.
        assert_eq!(
            p.map_target(p.guest_view, TargetClass::ScreenRelative),
            Extent::new(3440, 1440)
        );
        assert_eq!(
            p.classify(Extent::new(1290, 540)),
            TargetClass::ScreenRelative
        );

        let p = plan(
            Extent::new(5120, 1440),
            InternalResolution::MatchOutput,
            AspectPolicy::Expand,
        );
        assert_eq!(p.scene_aspect, Ratio::new(32, 9));
        assert_eq!(p.guest_view, Extent::new(3840, 1080));
    }

    #[test]
    fn sixteen_ten_letterboxes_or_expands_vertically() {
        let p = plan(
            Extent::new(1920, 1200),
            InternalResolution::MatchOutput,
            AspectPolicy::Original,
        );
        assert_eq!(
            p.presentation,
            Rect {
                x: 0,
                y: 60,
                width: 1920,
                height: 1080
            }
        );
        let p = plan(
            Extent::new(1920, 1200),
            InternalResolution::MatchOutput,
            AspectPolicy::Expand,
        );
        assert_eq!(p.guest_view, Extent::new(1920, 1200));
        assert_eq!(p.scene, Extent::new(1920, 1200));
    }

    #[test]
    fn mapped_rects_tile_without_gaps() {
        let p = plan(
            Extent::new(1366, 768),
            InternalResolution::MatchOutput,
            AspectPolicy::Original,
        );
        // 1366x768 is not exactly 16:9; the original aspect keeps 1365 columns.
        assert_eq!(p.scene, Extent::new(1365, 768));
        // Split the guest target into uneven columns and check seams.
        let cuts = [0, 333, 640, 961, 1500, 1920];
        let mapped: Vec<Rect> = cuts
            .windows(2)
            .map(|w| {
                p.map_rect(Rect {
                    x: w[0],
                    y: 0,
                    width: (w[1] - w[0]) as u32,
                    height: 1080,
                })
            })
            .collect();
        for pair in mapped.windows(2) {
            assert_eq!(pair[0].x + pair[0].width as i32, pair[1].x);
        }
        let last = mapped.last().unwrap();
        assert_eq!(last.x + last.width as i32, 1365);
        assert!(mapped.iter().all(|r| r.height == 768));
        // Off-screen origins keep their sign.
        let r = p.map_rect(Rect {
            x: -30,
            y: -30,
            width: 60,
            height: 60,
        });
        assert!(r.x < 0 && r.y < 0);
    }

    #[test]
    fn fixed_targets_keep_guest_size() {
        let p = plan(
            Extent::new(3840, 2160),
            InternalResolution::MatchOutput,
            AspectPolicy::Original,
        );
        assert_eq!(p.classify(Extent::new(2048, 2048)), TargetClass::Fixed);
        assert_eq!(
            p.map_target(Extent::new(2048, 2048), TargetClass::Fixed),
            Extent::new(2048, 2048)
        );
        assert_eq!(
            p.classify(Extent::new(240, 135)),
            TargetClass::ScreenRelative
        );
        assert_eq!(
            p.map_target(Extent::new(240, 135), TargetClass::ScreenRelative),
            Extent::new(480, 270)
        );
    }

    #[test]
    fn change_impact_classification() {
        let a = plan(
            Extent::new(2560, 1440),
            InternalResolution::Height(1440),
            AspectPolicy::Original,
        );
        let window_resized = plan(
            Extent::new(3000, 1688),
            InternalResolution::Height(1440),
            AspectPolicy::Original,
        );
        assert_eq!(
            a.impact_of_change(&window_resized),
            ChangeImpact::Presentation
        );
        let rescaled = plan(
            Extent::new(2560, 1440),
            InternalResolution::Height(1080),
            AspectPolicy::Original,
        );
        assert_eq!(a.impact_of_change(&rescaled), ChangeImpact::SceneTargets);
        assert_eq!(a.impact_of_change(&a.clone()), ChangeImpact::None);
    }

    #[test]
    fn rejects_degenerate_and_oversized_extents() {
        let req = |output, internal| ResolutionRequest {
            guest_logical: GUEST,
            output,
            internal,
            aspect: AspectPolicy::Original,
            max_dimension: Some(16384),
        };
        assert!(
            ResolutionPlan::new(&req(Extent::new(0, 720), InternalResolution::MatchOutput))
                .is_err()
        );
        assert!(
            ResolutionPlan::new(&req(
                Extent::new(1280, 720),
                InternalResolution::Height(20000)
            ))
            .is_err()
        );
        // A one-pixel-tall window still yields a valid plan.
        let p = ResolutionPlan::new(&req(Extent::new(1280, 1), InternalResolution::MatchOutput))
            .unwrap();
        assert_eq!(p.scene.height, 1);
    }
}
