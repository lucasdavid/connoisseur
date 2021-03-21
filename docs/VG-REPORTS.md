# training: VGDB-2016

VGDB-2016/test
  patch:
     vg: 25 * 50, 95%
    nvg: 42 * 50, 87%

  painting:
     vg: 25, 100%
    nvg: 42, 93%

RMP/test
  patch:
     vg: 55 * 50, 66%
    nvg: 85 * 50, 91%

  painting:
     vg: 55, 67%
    nvg: 85, 96%

RMP16/test: 
  patch:
     vg: 50 * (25 + 55), (25*50 * 95% + 55*50 * 66%)/(50*(25+55)) = 75.0625%
    nvg: 50 * (42 + 85), (42*50 * 87% + 85*50 * 91%)/(50*(42+85)) = 89.6771654%
    
    class-balanced acc: 82.3698327%

  painting:
     vg: 25 + 55, (25*100 + 55*67) / 80 = 77,3125%
    nvg: 42 + 85, (42*93 + 85*96) / 127 = 95,007874016%

    class-balanced acc: 86,160187008%


# training: RMP16

VGDB-2016/test
  patch:
     vg: 25 * 50, 97%
    nvg: 42 * 50, 86%

  painting:
     vg: 25, 100%
    nvg: 42, 93%

RMP/test
  patch:
     vg: 55 * 50, 68%
    nvg: 85 * 50, 89%

  painting:
     vg: 55, 70%
    nvg: 85, 95%

RMP16/test: 
  patch:
     vg: 50 * (25 + 55), (25*50 * 97 + 55*50 * 68) / (50*80) = 77,0625%
    nvg: 50 * (42 + 85), (42*50 * 86 + 85*50 * 89) / 6350 = 88,007874016%

    class-balanced acc: 82,54%
  
  painting:
     vg: 55, (25*100 + 55*70) / 80 = 79,375%
    nvg: 85, (42*93 + 85*95) / 127 = 94,338582677%

    class-balanced acc: 86,86%
