#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)

if (length(args) == 0) {
  stop("Must provide the folder to operate on", call. = FALSE)
}

library(tidyverse)
setwd(args[1])
files <- dir(pattern = "*.csv") %>% str_subset("loss", negate = T)
names(files) <- files
data <- map_df(files, read_csv, .id = "metric") %>%
  mutate(metric = str_remove(metric, ".csv")) %>%
  pivot_wider(names_from = metric, values_from = value) %>%
  mutate(train_f1 = 2 * train_tp / (2 * train_tp + train_fp + train_fn), val_f1 = 2 * val_tp / (2 * val_tp + val_fp + val_fn)) %>%
  mutate(selected_epoch = (val_acc == max(val_acc) & cumsum(val_acc == max(val_acc)) == 1))
data %>% write_tsv("metrics.tsv")

# data %>%
# filter(selected_epoch) %>%
# select(val_tp, val_fp, val_fn, val_tn) %>%
# pivot_longer(names_to = "metric", values_to = "value", everything()) %>%
# mutate(pred_p = str_detect(metric, "p"), real_p = metric %in% c("val_tp", "val_fn")) %>%
# ggplot(aes(x = real_p, y = pred_p, fill = value)) +
# geom_tile() +
# geom_text(aes(label = value)) +
# scale_fill_distiller(palette = "Blues", direction = 1)
