def makeRegistrar():
  """
  Registers decorator
  """
  registry = {}
  def registrar(func):
     registry[func.__name__] = func
     return func  # normally a decorator returns a wrapped function,
                  # but here we return func unmodified, after registering it
  registrar.all = registry
  return registrar


global sensorCallback
global rosPublisherCreator
sensorCallback = makeRegistrar()
rosPublisherCreator = makeRegistrar()
