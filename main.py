import pygame as interface
import pymunk
import pymunk.pygame_util
from agent import Agent
from geneticAlgorithm import Genetic_algorithm 

class Main():
    def __init__(self, number_of_agents):        
        self.screen = interface.display.set_mode((600, 600), interface.DOUBLEBUF)
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        self.draw_options.flags = pymunk.SpaceDebugDrawOptions.DRAW_SHAPES
        self.draw_options.flags |= pymunk.SpaceDebugDrawOptions.DRAW_CONSTRAINTS
        pymunk.pygame_util.positive_y_is_up = True
        self.clock = interface.time.Clock()
        self.running = True
        self.dead_agents = 0
        self.agents = [None] * number_of_agents        
        self.worlds = [pymunk.Space() for _ in range(number_of_agents)]  # Separate space for each agent
        
        for i in range(number_of_agents):
            self.worlds[i].gravity = (0.0, -981.0)
        
        position_of_agent = (0.5 * interface.display.get_window_size()[0], 150)

        for i in range(0, number_of_agents):
            self.agents[i] = Agent(position_of_agent, self.worlds[i])  # Pass a different world to each agent
        
        Agent.plateau_is_initialized = False
            
        number_of_weights = self.agents[0].neural_net.get_number_of_weights()
        number_of_biases = self.agents[0].neural_net.get_number_of_biases()
        self.genetic_algorithm = Genetic_algorithm(number_of_agents, number_of_weights, number_of_biases)
		
        for i in range(number_of_agents):
            self.agents[i].neural_net.set_weights(self.genetic_algorithm.population[i].weights)
            self.agents[i].neural_net.set_biases(self.genetic_algorithm.population[i].biases)
	
    def run(self):	
        while self.running:
            for event in interface.event.get():
                if event.type == interface.QUIT:
                    self.running = False
                    interface.display.quit()
                    return
                if event.type == interface.MOUSEBUTTONDOWN:
                    print('Forcing a new generation')
                    for agent in self.agents:
                        agent.is_alive = False
                        agent.destroy()
                        self.dead_agents = len(self.agents)
                        
            self.update()	
            self.draw()

    def update(self):
        for agent in self.agents:
            if agent.update():
                self.dead_agents += 1
        
        if(self.dead_agents == len(self.agents)):
            self.genetic_algorithm.update(self.agents)
            self.genetic_algorithm.upgrade()
            self.dead_agents = 0
            for i, agent in enumerate(self.agents):
                agent.reset()
                agent.neural_net.set_weights(self.genetic_algorithm.population[i].weights)
                agent.neural_net.set_biases(self.genetic_algorithm.population[i].biases)
                        
        # Update the physics engine for each world
        for world in self.worlds:
            world.step(0.01)   # I this high number, because the learning process goes too slow otherwise     

    def draw(self):
        self.clock.tick(400)
        interface.display.set_caption(f'FPS: {self.clock.get_fps() :.0f}')

        self.screen.fill(interface.Color('white')) 
        
        # Draw the best agent
        i = 0
        for agent in self.agents:
            if not agent.is_alive:
                continue
            agent.draw(interface, self.screen)
            i += 1
            if i > 0:
                break

        # Draw plateau
        Agent.draw_shapes([Agent.shape_plateau], interface, self.screen)        
        interface.display.flip()

print('\014')
main = Main(2**7)
main.run()