library(shiny)
library(leaflet)

# Define UI for application that draws a histogram
shinyUI(
  navbarPage("Bike Station APP", id = "inTabset",

    tabPanel(title="Home", value="home",
      fluidPage(
        # Application title
        titlePanel("Welcome in the 'BIKE STATION APP' !"),br(),
        
        # Sidebar with a slider input for number of bins 
        fluidRow(
          column(1, align="center"),
          column(5, align="center",
            radioButtons("opt_choice", label = "Choose a city",
              choices = list("Toulouse" = 1, "Dublin" = 2, "Lyon" = 3, "Nantes" = 4, "Marseille" = 5, "Stockholm" = 6, "Luxembourg" = 7),
              selected = 1)
          ),
          column(5, align="center",
            leafletOutput("map")
          ),
          column(1, align="center")
        )
      )
    )
  )
)

