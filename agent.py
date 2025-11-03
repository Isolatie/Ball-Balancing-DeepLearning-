import numpy as np
import pymunk
from neuralNetwork import Neural_network

class Agent():	
    # Static fields
    plateau_is_initialized = False
    time_no_disturb = 1000
    
    # Convenience function to transform the (x, +y) system to Pygame's (x, -y) coordinate system        
    @staticmethod
    def x2x_y2miny(v, interface):
        return v.x, interface.display.get_window_size()[1] - v.y
    
    # Draw pymunk shapes with pygame    
    @staticmethod
    def draw_shapes(shapes, interface, screen):
        for shape in shapes: 
            v = shape.body.position
            if isinstance(shape, pymunk.shapes.Poly):  # Draw Polygon
                V = [v_poly.rotated(shape.body.angle) + v for v_poly in shape.get_vertices()]              
                V = list(map(lambda v: Agent.x2x_y2miny(v, interface), V))                
                interface.draw.polygon(screen, shape.color, V, 0)
                interface.draw.aalines(screen, interface.Color('black'), True, V)
            elif isinstance(shape, pymunk.shapes.Circle):  # Draw Circle
                position = Agent.x2x_y2miny(v, interface)
                interface.draw.circle(screen, shape.color, position, shape.radius)
                interface.draw.circle(screen, interface.Color('black'), position, shape.radius, 1)  # Draw border for circle

    
    def __init__(self, position_of_agent, world):

        # Static fields, the plateau is a common field to share by all agents
        if not Agent.plateau_is_initialized:
            Agent.width_plateau = 500
            Agent.height_plateau = 15
            Agent.body_plateau = pymunk.Body(body_type = pymunk.Body.STATIC)    
            Agent.body_plateau.position = (position_of_agent[0], position_of_agent[1] - 0.5 * Agent.height_plateau)
            Agent.shape_plateau = pymunk.Poly.create_box(Agent.body_plateau, (Agent.width_plateau, Agent.height_plateau))
            Agent.shape_plateau.friction = 0.1
            Agent.shape_plateau.color = (0, 255, 0, 255)
            world.add(Agent.body_plateau, Agent.shape_plateau)            
            Agent.plateau_is_initialized = True
            
        # Parameters for agent
        self.position_of_agent = position_of_agent     
        self.width_cart        = 150
        self.height_cart       = 25
        self.mass_cart         = 1

        self.radius_ball       = 20
        self.mass_ball         = 0.1
        self.max_force         = 50
        self.angle_treshold    = np.radians(15)
        self.neural_net        = Neural_network([6, 1, 1])
        self.inputs            = [None] * self.neural_net.number_of_inputs
        self.world             = world
        self.reset()

    def reset(self):
        self.is_alive        = True
        self.fitness         = 0
        self.time_no_disturb = Agent.time_no_disturb     
        self.shapes          = []
        self.joints          = []
        self.x_cart = 0
        self.x_dot_cart = 0

        self.x1 = 0
        self.y1 = 0
        self.x1_dot = 0   
        self.y1_dot = 0

        self.x2 = 0
        self.y2 = 0
        self.x2_dot = 0    
        self.y2_dot = 0
        self.create_agent()

    def create_agent(self):
        # Cart
        body_cart = pymunk.Body(body_type = pymunk.Body.DYNAMIC)
        body_cart.position = (self.position_of_agent[0], self.position_of_agent[1] + 0.5 * self.height_cart)
        shape_cart = pymunk.Poly.create_box(body_cart, (self.width_cart, self.height_cart))
        shape_cart.collision_type = 1
        shape_cart.filter = pymunk.ShapeFilter(group=1)
        shape_cart.mass = self.mass_cart
        shape_cart.friction = 1
        shape_cart.color = (0, 0, 255, 255)
        prismatic_joint_cart = pymunk.GrooveJoint(Agent.body_plateau, body_cart, 
                                                  (-0.5 * Agent.width_plateau, 0.5 * Agent.height_plateau), 
                                                  ( 0.5 * Agent.width_plateau, 0.5 * Agent.height_plateau),                                                                                
                                                  (0, -0.5 * self.height_cart))
 
        # Ball 1 (bottom ball)
        body_ball1 = pymunk.Body()
        body_ball1.position = (body_cart.position.x, body_cart.position.y + self.height_cart + self.radius_ball)
        shape_ball1 = pymunk.Circle(body_ball1, self.radius_ball)
        shape_ball1.mass = self.mass_ball
        shape_ball1.friction = 1.0
        shape_ball1.color = (255, 0, 0, 255)
        shape_ball1.collision_type = 2
        shape_ball1.filter = pymunk.ShapeFilter(group=2)

        # Ball 2 (top ball)
        body_ball2 = pymunk.Body()
        body_ball2.position = (body_cart.position.x, body_ball1.position.y + 2 * self.radius_ball)
        shape_ball2 = pymunk.Circle(body_ball2, self.radius_ball)
        shape_ball2.mass = self.mass_ball
        shape_ball2.friction = 1.0
        shape_ball2.color = (255, 0, 0, 255)
        shape_ball2.collision_type = 2
        shape_ball2.filter = pymunk.ShapeFilter(group=3)

        # Add bodies, shapes, and joints to the world
        self.world.add(shape_cart.body, shape_cart)
        self.world.add(prismatic_joint_cart)
        self.world.add(shape_ball1.body, shape_ball1)
        self.world.add(shape_ball2.body, shape_ball2)
        
        # Stack all shapes/joints in a list
        self.shapes = [shape_cart, shape_ball1, shape_ball2]
        self.joints = [prismatic_joint_cart]
        
    def destroy(self):
        try:
            for joint in self.joints:
                self.world.remove(joint)
            for shape in self.shapes:
                self.world.remove(shape, shape.body)
        except:
            pass            

    def disturb(self):
        self.time_no_disturb -= 1
        self.time_no_disturb = np.clip(self.time_no_disturb, 0, Agent.time_no_disturb)

        # Disturbs when the speed of the ball on top is very low or when the position of the ball is like 20% away from the edge of the cart
        if (abs(self.x2_dot) <= 0.01) or ((self.x2 <= self.x_cart - 0.8 * 0.5*self.width_cart) or (self.x2 >= self.x_cart + 0.8 * 0.5*self.width_cart)):
            sign_force = -np.sign(self.x2 - self.x_cart)  # Push in the opposite direction
            F = sign_force * np.random.uniform(0.1, 1)
            self.time_no_disturb = Agent.time_no_disturb
            shape_ball2 = self.shapes[2]
            shape_ball2.body.apply_impulse_at_local_point((F, 0), (0, 0))
            shape_ball2.color = (255, 250, 250, 255)  # Visual indicator


    def update(self):
        if not self.is_alive:
            return

        # Pass the input data
        shape_cart = self.shapes[0]
        shape_ball1 = self.shapes[1]
        shape_ball2 = self.shapes[2]

        # Get positions and velocities
        self.x_cart = shape_cart.body.position.x - self.position_of_agent[0]
        self.x_dot_cart = shape_cart.body.velocity.x

        self.x1 = shape_ball1.body.position.x - self.position_of_agent[0]
        self.y1 = shape_ball1.body.position.y - self.position_of_agent[1]
        self.x1_dot = shape_ball1.body.velocity.x    

        self.x2 = shape_ball2.body.position.x - self.position_of_agent[0]
        self.y2 = shape_ball2.body.position.y - self.position_of_agent[1]
        self.x2_dot = shape_ball2.body.velocity.x    

        # Check if the cart falls off the plateau
        if abs(self.x_cart) > 0.5 * Agent.width_plateau:
            self.is_alive = False
            self.destroy()
            return True

        # Check if the balls are too far away from each other
        if abs(shape_ball1.body.position.x - shape_ball2.body.position.x) > self.radius_ball * 3:
            self.is_alive = False
            self.destroy()
            return True

        # Check if either ball hits the ground
        if self.y1 <= self.height_cart or self.y2 <= (self.height_cart):
            self.is_alive = False
            self.destroy()
            return True

        # Update neural network
        self.inputs[0] = self.x_cart / (0.5 * Agent.width_plateau)
        self.inputs[1] = self.x_dot_cart / 250
        self.inputs[2] = self.x1 / (0.5 * Agent.width_plateau)
        self.inputs[3] = self.x1_dot / 250
        self.inputs[4] = self.x2 / (0.5 * Agent.width_plateau)
        self.inputs[5] = self.x2_dot / 250

        # Compute the output of the neural network
        outputs = self.neural_net.update(self.inputs)
        F = 2 * (outputs[0] - 0.5) * self.max_force
        shape_cart.body.apply_impulse_at_local_point((F, 0), (0, -0.5 * self.height_cart))

        # Set a constraint on the rotation of the cart
        shape_cart.body.angle = 0
        self.fitness += 1

        # Set state of color
        fraction = self.time_no_disturb / Agent.time_no_disturb
        shape_ball1.color = (shape_ball1.color[0], fraction * shape_ball1.color[1], fraction * shape_ball1.color[2], 255) if shape_ball1 in self.shapes else (0, 0, 0, 0)
        shape_ball2.color = (shape_ball2.color[0], fraction * shape_ball2.color[1], fraction * shape_ball2.color[2], 255) if shape_ball2 in self.shapes else (0, 0, 0, 0)

        self.disturb()
        
    def draw(self, interface, screen):
        Agent.draw_shapes(self.shapes, interface, screen)