import UI.app as app
import AI.playground.first_steps_data as fsd


if __name__ == "__main__":
    my_app = app.StockPredictorApp()  
    my_app.run()                      
    fsd.closing_data(my_app)  
    
   
