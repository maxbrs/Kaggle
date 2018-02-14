#debut <- Sys.time()

library(shiny)
library(RMySQL)
library(leaflet)
library(ggplot2)
library(ggmap)

# setwd("C:/Users/mbriens/Documents/Kaggle/GIT/Kaggle/Projet-Bikes-JCDecaux")

#----------
# Connecting database :
#----------

user = 'root'
password = 'admin'
database_name = 'bikestations'
host = 'localhost'
port = 5005

mydb = dbConnect(MySQL(), user=user, password=password, dbname=database_name, host=host, port=port)
query = dbSendQuery(mydb, "SELECT * FROM STATION")
data = fetch(query, n=-1)
list_cities = c("Toulouse", "Dublin", "Lyon", "Nantes", "Marseille", "Stockholm", "Luxembourg")

# Define server logic required to draw a histogram
shinyServer(function(input, output) {

  output$map <- renderLeaflet({
    res = as.numeric(input$opt_choice)
    city = list_cities[res]
    df <- data[data$sta_city == city,]
    
    #coord <- geocode(city)
    coord = data.frame(lat = mean(df$sta_lat), lon = mean(df$sta_lon))

    plt <- leaflet(df) %>% setView(lng = coord$lon, lat = coord$lat, zoom = 12) %>%
      addProviderTiles(providers$Stamen, options = providerTileOptions(opacity = 0.25)) %>%
      addProviderTiles(providers$Stamen.TonerLabels) %>%
      addMarkers(~sta_lon, ~sta_lat, label = ~sta_name)
    print(plt)
  })
  
})
