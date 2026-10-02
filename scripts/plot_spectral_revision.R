#!/usr/bin/env Rscript
# Style-only inheritance from plot_spectral_manuscript.R; all statistics use revised tables.
library(ggplot2)
library(patchwork)
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) stop("Usage: Rscript plot_spectral_revision.R EVIDENCE OUTPUT")
evidence <- args[1]
output <- args[2]
dir.create(output, recursive = TRUE, showWarnings = FALSE)
read_table <- function(name) read.csv(file.path(evidence, name), check.names = FALSE)
theme_set(theme_classic(base_size = 8, base_family = "Arial") + theme(
  axis.line = element_line(linewidth = .3), axis.ticks = element_line(linewidth = .3),
  legend.title = element_blank(), legend.text = element_text(size = 7),
  strip.text = element_text(size = 8), plot.tag = element_text(face = "bold", size = 9),
  plot.title = element_text(size = 8), panel.grid = element_blank()))
palette <- c(H1 = "#777777", G1 = "#B57A43", C = "#246C85")
line_styles <- c(H1 = "dashed", G1 = "dotdash", C = "solid")
save_figure <- function(p, stem, height_mm) {
  width_mm <- 183
  svglite::svglite(file.path(output, paste0(stem, ".svg")), width = width_mm/25.4, height = height_mm/25.4)
  print(p); dev.off()
  grDevices::cairo_pdf(file.path(output, paste0(stem, ".pdf")), width = width_mm/25.4, height = height_mm/25.4, family = "Arial")
  print(p); dev.off()
  ragg::agg_png(file.path(output, paste0(stem, ".png")), width = width_mm, height = height_mm, units = "mm", res = 600)
  print(p); dev.off()
  ragg::agg_tiff(file.path(output, paste0(stem, ".tiff")), width = width_mm, height = height_mm, units = "mm", res = 600, compression = "lzw")
  print(p); dev.off()
}
ar_low <- 11.54835442
ar_high <- 11.82807116
d <- read_table("distribution_reference.csv")
concentration <- read_table("concentration.csv")
final <- subset(d, scope == "final" & family == "C")
final$start <- factor(final$seed)
p_a <- ggplot(final, aes(energy_eV, density_per_eV, colour = start, linetype = start)) +
  annotate("rect", xmin = ar_low, xmax = ar_high, ymin = 0, ymax = Inf, fill = "#B5B5B5", alpha = .3) +
  geom_line(linewidth = .55) + scale_colour_manual(values = c("#246C85", "#B57A43", "#776C8E")) +
  labs(x = "Effective energy (eV)", y = "Density (1/eV)", title = "Final C: three initializations")
outer <- read_table("outer_scores.csv")
med <- aggregate(nrmse ~ heldout_vr + family, outer, median)
med$lower <- aggregate(nrmse ~ heldout_vr + family, outer, min)$nrmse
med$upper <- aggregate(nrmse ~ heldout_vr + family, outer, max)$nrmse
p_b <- ggplot(med, aes(factor(heldout_vr), nrmse, colour = family, shape = family)) +
  geom_linerange(aes(ymin = lower, ymax = upper), position = position_dodge(.4), linewidth = .4) +
  geom_point(position = position_dodge(.4), size = 1.7) + scale_colour_manual(values = palette) +
  labs(x = "Held-out retarding voltage (V)", y = "Whole-curve NRMSE", title = "Prediction: median and start range")
modes <- subset(concentration, family == "C" & !grepl("^profile/", scope))
modes$scope <- factor(modes$scope, levels = c("outer_0V", "outer_4V", "outer_6V", "outer_8V", "final"),
                      labels = c("Holdout 0 V", "Holdout 4 V", "Holdout 6 V", "Holdout 8 V", "Final"))
modes$peak_status <- factor(modes$peak_status, levels = c("concentrated", "diffuse_spectrum", "boundary_peak"))
modes$row <- as.integer(modes$scope) + .12 * (modes$seed - 1)
p_c <- ggplot(modes, aes(mode_eV, row, shape = peak_status, colour = factor(seed))) +
  annotate("rect", xmin = ar_low, xmax = ar_high, ymin = -Inf, ymax = Inf, fill = "#B5B5B5", alpha = .3) +
  geom_point(size = 2) + scale_shape_manual(values = c(16, 1, 17),
    labels = c("Interior peak", "Diffuse", "Boundary")) +
  scale_y_continuous(breaks = 1:5, labels = levels(modes$scope)) +
  scale_colour_manual(values = c("#246C85", "#B57A43", "#776C8E")) +
  coord_cartesian(xlim = c(9, 16)) + labs(x = "Effective mode (eV)", y = NULL, title = "Mode stability and peak qualification")
recovery <- read_table("matched_recovery_seed_medians.csv")
recovery$truth <- factor(recovery$truth, levels = c("single", "gaussian", "broad", "asymmetric"),
                         labels = c("Delta", "Narrow", "Broad", "Asymmetric"))
truth_width <- unique(recovery[,c("truth", "truth_sd_eV")])
recovery$x <- as.integer(recovery$truth) + ifelse(recovery$kernel == "free", -.14, .14) + .025 * (recovery$replicate - 2)
p_d <- ggplot(recovery, aes(x, sd_eV, colour = kernel, shape = kernel)) +
  geom_point(size = 1.6, alpha = .7) +
  geom_point(data = truth_width, aes(as.integer(truth), truth_sd_eV), inherit.aes = FALSE, shape = 4, size = 2.5, stroke = .7) +
  scale_x_continuous(breaks = 1:4, labels = levels(recovery$truth)) +
  scale_colour_manual(values = c(free = "#B57A43", fixed = "#246C85")) +
  labs(x = "Synthetic generating density", y = "Recovered standard deviation (eV)", title = "Matched 644-point recovery; crosses: truth")
save_figure((p_a + p_b) / (p_c + p_d) + plot_annotation(tag_levels = "a"), "fig8_spectral_revision", 142)

points <- read_table("real_prediction_points.csv")
training <- subset(points, scope == "final_training")
pred <- aggregate(predicted_uA ~ Vr + Va + family, training, median)
observed <- unique(training[,c("Vr", "Va", "observed_uA")])
p_train <- ggplot() + geom_point(data = observed, aes(Va, observed_uA), size = .45, colour = "#333333") +
  geom_line(data = pred, aes(Va, predicted_uA, colour = family, linetype = family), linewidth = .45) +
  facet_wrap(~ Vr, ncol = 2, labeller = label_both) + scale_colour_manual(values = palette) +
  scale_linetype_manual(values = line_styles) +
  labs(x = "Accelerating voltage (V)", y = "Current (microampere)", title = "Revised final fits: four training conditions")
save_figure(p_train, "fig9_revision_training", 118)
stress <- subset(points, scope == "stress_10V")
pred <- aggregate(predicted_uA ~ Va + family, stress, median)
observed <- unique(stress[,c("Va", "observed_uA")])
p_stress <- ggplot() + geom_point(data = observed, aes(Va, observed_uA), size = .6, colour = "#333333") +
  geom_line(data = pred, aes(Va, predicted_uA, colour = family, linetype = family), linewidth = .6) + scale_colour_manual(values = palette) +
  scale_linetype_manual(values = line_styles) +
  labs(x = "Accelerating voltage (V)", y = "Current (microampere)", title = "Unchanged 10 V: frozen-selection stress test")
save_figure(p_stress, "fig10_revision_stress", 75)
save_figure((p_train / p_stress) + plot_layout(heights = c(2, 1)) + plot_annotation(tag_levels = "a"),
            "fig11_revision_curves", 166)

profiles <- subset(d, grepl("^profile/", scope) & family == "C")
profile_labels <- c(coarse_knots = "Knots: 0.25 eV", expanded_domain = "Domain: 8-17 eV",
                    fine_knots = "Knots: 0.05 eV", width_1 = "Width: 1 V",
                    width_2 = "Width: 2 V", width_3 = "Width: 3 V")
profiles$profile <- unname(profile_labels[sub("profile/", "", profiles$scope)])
profiles <- aggregate(density_per_eV ~ profile + energy_eV, profiles, median)
p_profiles <- ggplot(profiles, aes(energy_eV, density_per_eV)) +
  annotate("rect", xmin = ar_low, xmax = ar_high, ymin = 0, ymax = Inf, fill = "#B5B5B5", alpha = .3) +
  geom_line(colour = palette["C"], linewidth = .55) + facet_wrap(~profile, ncol = 3) +
  labs(x = "Effective energy (eV)", y = "Pointwise median density (1/eV)", title = "Fresh response-width, knot and domain refits")
save_figure(p_profiles, "figS6_revision_sensitivity", 108)
band <- read_table("bootstrap_density_band.csv")
p_boot <- ggplot(band, aes(energy_eV, median_density)) +
  annotate("rect", xmin = ar_low, xmax = ar_high, ymin = 0, ymax = Inf, fill = "#B5B5B5", alpha = .3) +
  geom_ribbon(aes(ymin = lower_density, ymax = upper_density), fill = palette["C"], alpha = .2) +
  geom_line(colour = palette["C"], linewidth = .55) +
  labs(x = "Effective energy (eV)", y = "Density (1/eV)", title = "30 conditional block resamples; median over starts per replicate")
save_figure(p_boot, "figS7_revision_bootstrap", 75)
