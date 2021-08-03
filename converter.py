import argparse
import re

argsdict = [
	{
		"shorthand": "-i",
		"longhand": "--infile",
		"helpmessage": "<-| Help |->"
	}
]
# Parse the Arguments
parser = argparse.ArgumentParser(description='Convert Data Storage to Alternate Format')

def addarg(params, prsr):
	prsr.add_argument(params["shorthand"], params["longhand"], help = params["helpmessage"])

for arg in argsdict:
	addarg(arg, parser)

args = vars(parser.parse_args())
inputextension = re.sub(".*(?=\\.)\\.", "", args["infile"])


print(inputextension)