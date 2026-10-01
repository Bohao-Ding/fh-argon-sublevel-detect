#!/usr/bin/env Rscript
# Quantitative panels from frozen evidence; no fitted or simulated observations are invented here.
library(ggplot2)
library(patchwork)
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) stop("Usage: Rscript plot_spectral_manuscript.R EVIDENCE OUTPUT")
evidence <- args[1]
output <- args[2]
dir.create(output, recursive = TRUE, showWarnings = FALSE)
read_table <- function(name) read.csv(file.path(evidence, name), check.names = FALSE)
theme_set(theme_classic(base_size = 8, base_family = "Arial") + theme(
  axis.line = element_line(linewidth = 0.3), axis.ticks = element_line(linewidth = 0.3),
  legend.title = element_blank(), legend.text = element_text(size = 7),
  strip.text = element_text(size = 8), plot.tag = element_text(face = "bold", size = 9),
  plot.title = element_text(size = 8), panel.grid = element_blank()))
palette <- c(H1 = "#777777", G1 = "#B57A43", C = "#246C85")
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
densities <- read_table("distribution_reference.csv")
continuous <- subset(densities, scope == "final" & family == "C")
median_q <- aggregate(density_per_eV ~ family + energy_eV, continuous, median)
boot <- subset(densities, grepl("^bootstrap_", scope) & family == "C")
interval <- aggregate(density_per_eV ~ energy_eV, boot, function(x) quantile(x, c(.025, .975)))
interval$lower <- interval$density_per_eV[,1]
interval$upper <- interval$density_per_eV[,2]
nist <- data.frame(energy = c(11.54835442, 11.62359272, 11.72316039, 11.82807116))
delta <- subset(densities, scope == "final" & family == "H1")
p_a <- ggplot() + geom_ribbon(data = interval, aes(energy_eV, ymin = lower, ymax = upper), fill = palette["C"], alpha = .18) +
  geom_vline(data = nist, aes(xintercept = energy), colour = "#B9B9B9", linetype = "dotted", linewidth = .4) +
  geom_line(data = continuous, aes(energy_eV, density_per_eV, group = interaction(family, seed), colour = family), alpha = .3, linewidth = .35) +
  geom_line(data = median_q, aes(energy_eV, density_per_eV, colour = family), linewidth = .6) +
  geom_rug(data = delta, aes(x = energy_eV), sides = "b", colour = palette["H1"]) +
  scale_colour_manual(values = palette) + labs(x = "Effective energy (eV)", y = "Density (1/eV)", title = "Final continuous distribution")
outer <- read_table("outer_scores.csv")
coarse <- subset(outer, family %in% names(palette))
score_median <- aggregate(nrmse ~ heldout_vr + family, coarse, median)
score_min <- aggregate(nrmse ~ heldout_vr + family, coarse, min)
score_max <- aggregate(nrmse ~ heldout_vr + family, coarse, max)
score_median$lower <- score_min$nrmse
score_median$upper <- score_max$nrmse
p_b <- ggplot(score_median, aes(factor(heldout_vr), nrmse, colour = family)) +
  geom_linerange(aes(ymin = lower, ymax = upper), position = position_dodge(.45), linewidth = .45) +
  geom_point(position = position_dodge(.45), size = 1.8) + scale_colour_manual(values = palette) +
  labs(x = "Held-out retarding voltage (V)", y = "Whole-curve NRMSE", title = "Prediction: median and start range")
profile <- subset(densities, grepl("^final/width_", scope) & family == "C")
profile$width <- sub("final/width_", "", profile$scope)
profile <- aggregate(density_per_eV ~ width + energy_eV, profile, median)
p_c <- ggplot(profile, aes(energy_eV, density_per_eV, colour = width)) + geom_line(linewidth = .6) +
  scale_colour_manual(values = c("1" = "#337F70", "2" = "#B78B47", "3" = "#7A6D8C"), labels = c("1 V", "2 V", "3 V")) +
  labs(x = "Effective energy (eV)", y = "Density (1/eV)", title = "Fixed-width refits")
concentration <- subset(read_table("concentration.csv"), scope == "final" & family == "C")
half_low <- median(concentration$peak_halfheight_low_eV)
half_high <- median(concentration$peak_halfheight_high_eV)
half_height <- max(median_q$density_per_eV)/2
p_d <- ggplot(median_q, aes(energy_eV, density_per_eV)) +
  annotate("rect", xmin = min(nist$energy), xmax = max(nist$energy), ymin = 0, ymax = Inf,
           fill = "#A5B6A4", alpha = .3) +
  geom_line(colour = palette["C"], linewidth = .6) +
  geom_vline(data = nist, aes(xintercept = energy), colour = "#889488", linetype = "dotted", linewidth = .35) +
  annotate("segment", x = half_low, xend = half_high, y = half_height, yend = half_height, linewidth = .6, colour = "#5E6770") +
  annotate("text", x = (half_low + half_high)/2, y = half_height - .05, label = "Main half-height interval", size = 2.5) +
  annotate("text", x = 11.69, y = .59, label = "First 4s group", size = 2.5) +
  coord_cartesian(xlim = c(11, 13), ylim = c(0, .64)) +
  labs(x = "Effective energy (eV)", y = "Density (1/eV)", title = "Main lobe and first 4s group")
save_figure((p_a + p_b) / (p_c + p_d) + plot_annotation(tag_levels = "a"), "fig6_spectral_inference", 132)
points <- read_table("real_prediction_points.csv")
training <- subset(points, scope == "final_training" & family %in% names(palette))
pred <- aggregate(predicted_uA ~ Vr + Va + family, training, median)
observed <- unique(training[,c("Vr", "Va", "observed_uA")])
p_train <- ggplot() + geom_point(data = observed, aes(Va, observed_uA), colour = "#333333", size = .45) +
  geom_line(data = pred, aes(Va, predicted_uA, colour = family), linewidth = .45) +
  facet_wrap(~ Vr, ncol = 2, labeller = label_both) + scale_colour_manual(values = palette) +
  labs(x = "Accelerating voltage (V)", y = "Current (microampere)", title = "Final fits: four training curves")
stress <- subset(points, scope == "stress_10V" & family %in% names(palette))
stress_pred <- aggregate(predicted_uA ~ Va + family, stress, median)
stress_obs <- unique(stress[,c("Va", "observed_uA")])
p_stress <- ggplot() + geom_point(data = stress_obs, aes(Va, observed_uA), colour = "#333333", size = .7) +
  geom_line(data = stress_pred, aes(Va, predicted_uA, colour = family), linewidth = .6) + scale_colour_manual(values = palette) +
  labs(x = "Accelerating voltage (V)", y = "Current (microampere)", title = "Unchanged 10 V: prediction after selection freeze")
save_figure(p_train / p_stress + plot_layout(heights = c(2.1, 1)) + plot_annotation(tag_levels = "a"), "fig7_spectral_curves", 152)
