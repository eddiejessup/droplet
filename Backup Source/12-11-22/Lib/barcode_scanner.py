rpc_key = '3a7a73eda4f48b1a3a1cb6b10a106d49a929af90'

class Error(Exception):
    pass
class BarcodeNotSupported(Error):
    pass
class BarcodeNotSeen(Error):
    pass
class LookupFailed(Error):
    pass

def image_to_barcode(fname):
    import zbar
    import Image
    scanner = zbar.ImageScanner()
    scanner.parse_config('enable')

    pil = Image.open(fname).convert('L')
    width, height = pil.size
    raw = pil.tostring()

    image = zbar.Image(width, height, 'Y800', raw)
    scanner.scan(image)
    data = [('%s' % symbol.type, symbol.data) for symbol in image]
    del(image)
    if len(data) == 0:
        raise BarcodeNotSeen
    return data

def video_to_barcode(device='/dev/video0'):
    import zbar
    def my_handler(proc, image, closure):
        return [('%s' % symbol.type, symbol.data) for symbol in image]

    proc = zbar.Processor()
    proc.parse_config('enable')
    proc.init(device)
    proc.set_data_handler(my_handler)
    proc.visible = True
    proc.active = True
    try:
        proc.user_wait()
    except zbar.WindowClosed:
        pass

def video_to_barcode_dummy(device='/dev/video0'):
    return ('ean,5000157024886')

def barcode_lookup(protocol, barcode):
    import xmlrpclib
    serv = xmlrpclib.ServerProxy('http://www.upcdatabase.com/xmlrpc')
    params = {'rpc_key': rpc_key}
    if protocol not in ['ean', 'upc']:
        raise BarcodeNotSupported('Barcode protocol %s is not supported.' % protocol)
    params[protocol] = barcode

    result = serv.lookup(params)
    if result['status'] == 'fail':
        raise LookupFailed(result['message'])
    return result
