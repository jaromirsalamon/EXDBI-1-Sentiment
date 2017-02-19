setwd("~/Dropbox/PhD/Workspace/EXDBI/assignment")
library(ggplot2)
model <- "w2v"
df <- read.csv(sprintf("data/out/%s_cnf.csv",model))

png(filename = sprintf("data/out/%s_cnf.png",model), width = 1440, height = 1620, units = "px", res = 180, bg = "white")
g <- ggplot(data =  df, mapping = aes(x = class.pred, y = class.act))
g <- g + geom_tile(aes(fill = cnf), colour = "darkgray")
g <- g + geom_text(aes(label = sprintf("%1.0f", cnf)), colour = "white", vjust = 1)
g <- g + scale_fill_gradient(low = "blue", high = "red")
g <- g + theme_minimal() + theme(legend.position = "none", axis.ticks = element_blank())
g <- g + labs(x = "Predicted class", y = "Actual class")
g <- g + scale_y_discrete(limits = rev(levels(df$class.act)))
g <- g + facet_wrap(~ accuracy, nrow = 3)
g <- g + scale_x_discrete(position = "top")
g
dev.off()