# Install packages
install.packages("gganimate")
library(tidyverse)
library(dplyr)
library(readr)
library(ggplot2)
library(gganimate)
install.packages("deldir")
library(deldir) 

# Load relevant data
games <- read_csv("games.csv")
player_play <- read_csv("player_play.csv")
players <- read_csv("players.csv")
plays <- read_csv("plays.csv")
tracking_week_2 <- read_csv("tracking_week_2.csv")

# Filter for chosen example play
example_play_567 <- tracking_week_2 %>%
  filter(playId == 567,
         gameId == 2022091802)

te_567_nflids <- player_play %>%
  filter(playId == 567,
         gameId == 2022091802,
         motionSinceLineset == 1) %>%
  select(nflId) %>%
  pull()

te_567 <- players %>%
  filter(nflId %in% te_567_nflids,
         position == "TE") %>%
  select(nflId) %>%
  pull()

# Taken from lane detect python script
play_567_defenders = c(46082.0, 47809.0, 47956.0, 53448.0)

animated_plot <- example_play_567 %>%
  ggplot(aes(x, y, color = club)) + 
  geom_point() + 
  geom_point(data = example_play_567 %>% filter(nflId == te_567), 
             aes(x, y), 
             color = "red", 
             size = 6, 
             shape = 1) + # Hollow circle
  geom_polygon(data = example_play_567 %>% filter(nflId %in% play_567_defenders),
               aes(x = x, y = y, group = frameId), # Use `frameId` to animate the polygon
               fill = "blue", 
               alpha = 0.4, 
               inherit.aes = FALSE) + # Shaded polygon with transparency
  scale_x_continuous(limits = c(0, 120),
                     breaks = seq(10, 110, by = 10),
                     labels = c("G", "10", "20", "30", "40", "50", "40", "30", "20", "10", "G"),
                     expand = c(0, 0)) +  
  scale_y_continuous(limits = c(0, 53.3), expand = c(0, 0)) +
  annotate("rect", xmin = 0, ymin = 0, xmax = 10, ymax = 53.3, fill = "grey", alpha = 0.5) + 
  annotate("rect", xmin = 110, ymin = 0, xmax = 120, ymax = 53.3, fill = "grey", alpha = 0.5) +
  theme(
    legend.title = element_blank()) +
  transition_states(
    frameId,  # Variable that controls the animation frames
    transition_length = 1,  # Duration of transitions between frames
    state_length = 1        # Duration each frame is shown
  ) +
  ease_aes('linear') +
  labs(title = "WAS vs. DET, Week 2, play 567",
       subtitle = 'Frame: {closest_state}')  

# Animate the plot with larger dimensions
animated_gif <- animate(
  animated_plot, 
  duration = 10, 
  fps = 10, 
  width = 600,   # Increase the width
  height = 400,  # Increase the height
  renderer = gifski_renderer()
)

# Save the animation
anim_save("example_play_567_animation.gif", animation = animated_gif)