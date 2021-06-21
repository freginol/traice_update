import time
import os

from pathlib import Path

class BatchStep:

    def __init__(self, input_files, pickled_dir, completion_file,cred_dict):

        self.input_files = input_files
        self.pickled_dir = pickled_dir
        self.completion_file = completion_file
        self.cred_dict=cred_dict

    def check_inputs(self):

        for file_name in self.input_files:

            f = Path(file_name)

            if not f.is_file():
                return False
        
        return True

    def check_completion_file_exists(self):

        f = Path(self.completion_file)

        if f.is_file():
            return True
        
        return False

    def run(self):

        if self.check_completion_file_exists():
            print('Skipping batch step', self.__class__.__name__, 
                'because completion file', self.completion_file,
                'is already present. To enable execution of this step, delete this completion file.')
            return

        assert (self.check_inputs()), ('Error: some inputs to batch step ' + 
            self.__class__.__name__ + ' are missing.')

        print('Running batch step:', self.__class__.__name__, '...')

        t_0 = time.time()
        self.run_step()
        t_1 = time.time()
        
        # Create completion file
        with open(self.completion_file, 'a'):
            os.utime(self.completion_file, None)

        print('Batch step', self.__class__.__name__, 'completed in', 
            round(t_1 - t_0, 3), 'seconds.')
    
    def run_step(self):
        pass
