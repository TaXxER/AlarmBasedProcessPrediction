library(ggplot2)

setwd("/home/irene/Repos/AlarmBasedProcessPrediction/results_calibrated/")
files <- list.files()

data <- data.frame()
for (file in files) {
  tmp <- read.table(file, sep=";", header=T)
  data <- rbind(data, tmp)  
}
data$method <- gsub("fixed", "", data$method)
data$metric <- gsub("_mean", "", data$metric)

head(data)

width <- 1300
height <- 900

# plot fscore and earliness
png("/home/irene/Dropbox/early_sequence_prediction_BPM_paper/images/earliness_fscore_fixed_threshold_baselines.png", width=width, height=height)
ggplot(subset(data, metric %in% c("fscore", "earliness")), aes(y=value, x=method, group=metric, fill=metric)) + 
  geom_bar(stat="identity", position="dodge") + theme_bw(base_size=32) +
  geom_text(aes(label=round(value, 2)),position = position_dodge(width=0.9),
            size=8) +
  facet_wrap( ~ dataset, scales="free", ncol=1)
dev.off()

# plot costs
data$cost_weights_str <- sprintf("no=%s\nfalse=%s\ntrue=%s", data$c01, data$c10, data$c11)

png("/home/irene/Dropbox/early_sequence_prediction_BPM_paper/images/costs_fixed_threshold_baselines_traffic_fines.png", width=width, height=height)
ggplot(subset(data, metric=="cost" & dataset=="traffic_fines_1"), aes(x=cost_weights_str, y=-value, group=method, fill=method)) + geom_bar(stat="identity", position="dodge", color="black") + 
  theme_bw(base_size=32) + ylab("reward") + xlab("cost weights for each alarm type") + facet_wrap( ~ dataset, scales="free", ncol=1)
dev.off()

png("/home/irene/Dropbox/early_sequence_prediction_BPM_paper/images/costs_fixed_threshold_baselines_bpic2012_declined.png", width=width, height=height)
ggplot(subset(data, metric=="cost" & dataset=="bpic2012_declined" & !(method %in% c("conf10", "conf20", "conf30", "conf90"))), aes(x=cost_weights_str, y=-value, group=method, fill=method)) + geom_bar(stat="identity", position="dodge", color="black") + 
  theme_bw(base_size=32) + ylab("reward") + xlab("cost weights for each alarm type") + facet_wrap( ~ dataset, scales="free", ncol=1)
dev.off()

png("/home/irene/Dropbox/early_sequence_prediction_BPM_paper/images/costs_fixed_threshold_baselines_bpic2012_cancelled.png", width=width, height=height)
ggplot(subset(data, metric=="cost" & dataset=="bpic2012_cancelled"), aes(x=cost_weights_str, y=-value, group=method, fill=method)) + geom_bar(stat="identity", position="dodge", color="black") + 
  theme_bw(base_size=32) + ylab("reward") + xlab("cost weights for each alarm type") + facet_wrap( ~ dataset, scales="free", ncol=1)
dev.off()

